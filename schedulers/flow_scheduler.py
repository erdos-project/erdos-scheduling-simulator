import os
import time
from copy import copy, deepcopy
from typing import Optional

import absl  # noqa: F401

import flowlessly_py
from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workers.workers import Worker, WorkerPool
from workload import Placement, Placements, Workload, Task
from workload.strategy import ExecutionStrategy

def select_best_execution_strategies(task: Task, worker: Worker) -> Optional[ExecutionStrategy]:
    """
    Selects the best execution strategy when assigning a task to a worker.
    
    Currently, this function selects the fastest strategy among the available ones.
    In Flowlessly, only a single edge can exist between two nodes, necessitating the selection
    of the best strategy among available ones.
    
    Args:
    - task (Task): The task for which the strategy needs to be selected.
    - worker (Worker): The worker on which the task would potentially be placed.

    Returns:
    - Optional[ExecutionStrategy]: The best execution strategy or None if no suitable strategy exists.
    """
    return worker.get_compatible_strategies(task.available_execution_strategies).get_fastest_strategy()

def populate_min_cost_max_flow_graph(flow_graph: flowlessly_py.AdjacencyMapGraph, tasks_to_be_scheduled: list[Task], schedulable_worker_pools: WorkerPools) -> tuple[dict[int, str], dict[int, str]]:
    """
    Populates the flow graph with nodes and edges to represent tasks, workers, and scheduling constraints.
    
    This function structures the graph such that the minimum cost maximum flow algorithm can determine
    the optimal task-to-worker assignments.

    TODO: Insert WorkerPool node between task and worker to reduce the amount of edges.
    Note: Batching is not supported as of now.
    
    Args:
    - flow_graph (flowlessly_py.AdjacencyMapGraph): The flow graph to be populated.
    - tasks_to_be_scheduled (list[Task]): List of tasks that need to be scheduled.
    - schedulable_worker_pools (WorkerPools): Worker pools containing workers on which tasks can be scheduled.

    Returns:
    - tuple[dict[int, str], dict[int, str]]: Mapping from node IDs to task IDs, and from node IDs to worker IDs.
    """
    # Create the global sink node.
    # Flowlessly declaration: uint32_t AddNode(int32_t supply, int64_t price, NodeType type, bool first_scheduling_iteration);
    node_id_global_sink = flow_graph.AddNode(-1, 0, flowlessly_py.NodeType.SINK, False)
    
    # Create a node that allows tasks to be unscheduled. 
    # Note: All tasks will have an edge toward this node.
    node_id_unscheduled = flow_graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)

    # Connect the unscheduled node to the global sink node with capacity equal to the totoal number of tasks
    # Flowlessly declaration: Arc* AddArc(uint32_t src_node_id, uint32_t dst_node_id, uint32_t min_flow, int32_t capacity, int64_t cost, int32_t type)
    flow_graph.AddArc(node_id_unscheduled, node_id_global_sink, 0, len(tasks_to_be_scheduled), 0, 0)

    # Create a node for each worker
    worker_id_to_node_id = {}
    node_id_to_worker_id = {}
    for worker_pool in schedulable_worker_pools.worker_pools:
        for worker in worker_pool.workers:
            node_id_woker = flow_graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)
            worker_id_to_node_id[worker.id] = node_id_woker
            node_id_to_worker_id[node_id_woker] = worker.id
            # For now assume one worker can only execute one task at a time. No support for batching as of now.
            flow_graph.AddArc(node_id_woker, node_id_global_sink, 0, 1, 0, 0)

    # Create a node for each task
    max_runtime = 0
    node_id_to_task_id = {}
    for task in tasks_to_be_scheduled:
        # Give this task node a supply of 1
        node_id_task = flow_graph.AddNode(1, 0, flowlessly_py.NodeType.OTHER, False)
        node_id_to_task_id[node_id_task] = task.id

        for worker_pool in schedulable_worker_pools.worker_pools:
            for worker in worker_pool.workers:
                # In Flowlessly, there can be only one edge between two nodes. So we need to somehow pick one best strategy from all the available ones.
                best_execution_strategy = select_best_execution_strategies(task, worker)
                if best_execution_strategy is not None:
                    # Flowlessly declaration: Arc* AddArc(uint32_t src_node_id, uint32_t dst_node_id, uint32_t min_flow, int32_t capacity, int64_t cost, int32_t type)
                    # TODO: Should the task deadline be somehow factored into edge cost?
                    flow_graph.AddArc(node_id_task, worker_id_to_node_id[worker.id], 0, 1, best_execution_strategy.runtime.time, 0)
                    max_runtime = max(max_runtime, best_execution_strategy.runtime.time)

    for node_id_task in node_id_to_task_id.keys():
        # Connect the task node to the unscheduled node to allow it to be unschedule
        # Set the cost of such edge to be twice the maximum runtime so that flow scheduler will prefer scheduling the task to run
        flow_graph.AddArc(node_id_task, node_id_unscheduled, 0, 1, max_runtime * 2, 0)
    
    return node_id_to_task_id, node_id_to_worker_id
    
def extract_placement_from_flow_graph(flow_graph: flowlessly_py.AdjacencyMapGraph, node_id_to_task_id: dict[int, str], node_id_to_worker_id: dict[int, str]) -> dict[str, str]:    
    """
    Extracts task-to-worker placements based on the solved flow graph.
    
    By examining the flow of tasks in the graph, this function determines which tasks have
    been assigned to which workers.
    
    Args:
    - flow_graph (flowlessly_py.AdjacencyMapGraph): The solved flow graph with scheduling information.
    - node_id_to_task_id (dict[int, str]): Mapping from node IDs to task IDs.
    - node_id_to_worker_id (dict[int, str]): Mapping from node IDs to worker IDs.

    Returns:
    - dict[str, str]: A mapping from task IDs to worker IDs.
    """
    task_id_to_worker_id: dict[str, str] = {}

    for node_id, task_id in node_id_to_task_id.items():
        for id_other_node, arc in flow_graph.get_arcs()[node_id].items():
            if arc.is_fwd:
                flow = arc.reverse_arc.residual_cap + arc.min_flow
                if flow > 0 and id_other_node in node_id_to_worker_id:
                    task_id_to_worker_id[task_id] = node_id_to_worker_id[id_other_node]

    return task_id_to_worker_id


class FlowScheduler(BaseScheduler):
    """Implements a flow-based scheduler based on the idea presented in 
    "Quincy: fair scheduling for distributed computing clusters" 
    (https://dl.acm.org/doi/10.1145/1629575.1629601).

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        if preemptive:
            raise ValueError("FlowScheduler does not support preemption.")
        super(FlowScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            time=sim_time,
            preemption=self.preemptive,
            worker_pools=worker_pools,
        )

        self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] Flow scheduler receive "
                f"{tasks_to_be_scheduled=}"
            )
        
        # TODO: How to handle preemption?
        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = copy(worker_pools)

        worker_id_to_worker: dict[str, Worker] = {}
        worker_id_to_worker_pool: dict[str, WorkerPool] = {}
        for worker_pool in schedulable_worker_pools.worker_pools:
            for worker in worker_pool.workers:
                worker_id_to_worker[worker.id] = worker
                worker_id_to_worker_pool[worker.id] = worker_pool
            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] The state of {worker_pool} "
                f"is:{os.linesep} {os.linesep.join(worker_pool.get_utilization())}"
            )

        scheduler_start_time = time.time()

        flow_stats = flowlessly_py.Statistics()
        flow_graph = flowlessly_py.AdjacencyMapGraph(flow_stats)
        
        node_id_to_task_id, node_id_to_worker_id = populate_min_cost_max_flow_graph(flow_graph, tasks_to_be_scheduled, schedulable_worker_pools)
        
        solver = flowlessly_py.SuccessiveShortest(flow_graph, flow_stats)
        solver.Run()
        
        task_id_to_worker_id = extract_placement_from_flow_graph(flow_graph, node_id_to_task_id, node_id_to_worker_id)

        placements = []
        for task in tasks_to_be_scheduled:
            if task.id not in task_id_to_worker_id:
                placements.append(Placement.create_task_placement(task=task))
                continue
            
            worker_id = task_id_to_worker_id[task.id]
            worker = worker_id_to_worker[worker_id]
            worker_pool = worker_id_to_worker_pool[worker_id]
            execution_strategy = select_best_execution_strategies(task, worker)

            worker.place_task(
                task, execution_strategy=execution_strategy
            )
            placements.append(
                Placement.create_task_placement(
                    task=task,
                    placement_time=sim_time,
                    worker_pool_id=worker_pool.id,
                    worker_id=worker_id,
                    execution_strategy=execution_strategy,
                )
            )
            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] Placed {task} on "
                f"Worker ({worker_id}) to be started at "
                f"{sim_time} with the execution strategy: "
                f"{execution_strategy}."
            )

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime if self.runtime == EventTime.invalid() else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )