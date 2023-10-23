import os
import time
from copy import copy, deepcopy
from operator import attrgetter
from typing import Optional

import absl  # noqa: F401

import flowlessly_py
from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Placement, Placements, Workload, Task


class FlowScheduler(BaseScheduler):
    """Implements a scheduler based on min cost max flow algorithm.

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

    def _populate_min_cost_max_flow_graph(self, flow_graph: flowlessly_py.AdjacencyMapGraph, tasks_to_be_scheduled: list[Task], schedulable_worker_pools: WorkerPools):
        # How to make sure that only one batch can be selected by the GPU?

        # Create the global sink node
        node_id_global_sink = flow_graph.AddNode(-1, 0, flowlessly_py.NodeType.OTHER, False)
        
        # Create a node that allows tasks to be unscheduled
        node_id_unscheduled = flow_graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)
        # Connect the unscheduled node to the global sink node with capacity equal to the totoal number of tasks
        flow_graph.AddArc(node_id_global_sink, node_id_unscheduled, 0, len(tasks_to_be_scheduled), 0, 0)

        # TODO: Create a node for each gpu? worker?
        for worker_pool in schedulable_worker_pools.worker_pools:
            for worker in worker_pool.workers:
                node_id_woker = flow_graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)
                # What should the capacity of this arc be?
                # flow_graph.AddArc(node_id_woker, node_id_global_sink, 0, ?, 0, 0)
                pass

        # Create node for each task
        for task in tasks_to_be_scheduled:
            for execution_strategy in task.available_execution_strategies:
                # in C++: AddNode(int32_t supply, int64_t price, NodeType type, bool first_scheduling_iteration)
                node_id_task = flow_graph.AddNode(1, 0, flowlessly_py.NodeType.OTHER, False)

                # Create a node for this execution_strategy if not already

                # Connect the task node to the execution_strategy node with capacity = 1, cost = 0

                # TODO: if can place this task on this worker/gpu, create an arc?
                # TODO: if the execution_strategy can be accomdate 
                for worker_pool in schedulable_worker_pools.worker_pools:
                    for worker in worker_pool.workers:
                        if worker.can_accomodate_strategy(execution_strategy):
                            # Arc* AddArc(uint32_t src_node_id, uint32_t dst_node_id, uint32_t min_flow, int32_t capacity, int64_t cost, int32_t type)
                            # Note that the Flowlessly librry expect the cost to be integer. Is this a problem?
                            # flow_graph.AddArc(execution_strategy_node_id, worker_id, 0, batch_size, latency // batch_size, 0)
                            pass

                # Connect the node to the unscheduled node
                flow_graph.AddArc(node_id_task, node_id_unscheduled, 0, 1, 0, 0)
                pass
    
    def _extract_placement_from_flow_graph(self, flow_graph: flowlessly_py.AdjacencyMapGraph) -> list[Placement]:    
        placements: list[Placement] = []
        # TODO:
        min_cost = 0
        for node_id in range(1, flow_graph.get_max_node_id() + 1):
            for id_other_node, arc in flow_graph.get_arcs()[node_id].items():
                if arc.is_fwd:
                    flow = arc.reverse_arc.residual_cap + arc.min_flow
                    if flow > 0:
                        print(f"flow from {node_id} to {id_other_node} is {flow} with cost {flow * arc.cost}")
                        min_cost += flow * arc.cost

        return placements
    
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

        print(f"{sim_time=}")
        for task in tasks_to_be_scheduled:
            print(f"{task.name=}, {task.task_graph=}, {task._timestamp=}, {task.deadline=}")
            print(f"execution strategies: {task.available_execution_strategies._strategies}")
        print()
        
        # TODO: How to handle preemptiveness?
        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = copy(worker_pools)

        for worker_pool in schedulable_worker_pools.worker_pools:
            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] The state of {worker_pool} "
                f"is:{os.linesep} {os.linesep.join(worker_pool.get_utilization())}"
            )

        scheduler_start_time = time.time()
        ##################### Flow #####################

        flow_stats = flowlessly_py.Statistics()
        flow_graph = flowlessly_py.AdjacencyMapGraph(flow_stats)
        
        self._populate_min_cost_max_flow_graph(tasks_to_be_scheduled, schedulable_worker_pools)
        
        solver = flowlessly_py.SuccessiveShortest(flow_graph, flow_stats)
        solver.Run()
        
        placements = self._extract_placement_from_flow_graph(flow_graph)

        ##################### Flow #####################
                
        ordered_tasks = list(
            sorted(
                tasks_to_be_scheduled, key=lambda item: (item.deadline, item.task_graph)
            )
        )

        task_descriptions = [
            f"{task.unique_name} ({task.deadline})" for task in ordered_tasks
        ]
        self._logger.debug(
            f"[{sim_time.to(EventTime.Unit.US).time}] The order of the "
            f"tasks is {task_descriptions}."
        )

        # Run the scheduling loop.
        # TODO (Sukrit): This loop may require spurious migrations of tasks
        # by preempting them from one pool, and assigning them to another.
        # We should ensure that we check if the worker pool already has running
        # tasks of lower priority, and only preempt the lowest priority one if
        # need be.
        placements = []
        for task in ordered_tasks:
            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] EDFScheduler trying to "
                f"schedule {task} with the available execution strategies: "
                f"{task.available_execution_strategies}."
            )
            is_task_placed = False
            for execution_strategy in task.available_execution_strategies:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    if worker_pool.can_accomodate_strategy(execution_strategy):
                        worker_pool.place_task(
                            task, execution_strategy=execution_strategy
                        )
                        is_task_placed = True
                        placements.append(
                            Placement.create_task_placement(
                                task=task,
                                placement_time=sim_time,
                                worker_pool_id=worker_pool.id,
                                execution_strategy=execution_strategy,
                            )
                        )
                        self._logger.debug(
                            f"[{sim_time.to(EventTime.Unit.US).time}] Placed {task} on "
                            f"Worker Pool ({worker_pool.id}) to be started at "
                            f"{sim_time} with the execution strategy: "
                            f"{execution_strategy}."
                        )
                        break
                if is_task_placed:
                    break

            if is_task_placed:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    self._logger.debug(
                        f"[{sim_time.to(EventTime.Unit.US).time}] The state of "
                        f"{worker_pool} is:{os.linesep}"
                        f"{os.linesep.join(worker_pool.get_utilization())}"
                    )
            else:
                self._logger.debug(
                    f"[{sim_time}] Failed to place {task} because no worker pool "
                    f"could accomodate the resource requirements."
                )
                placements.append(Placement.create_task_placement(task=task))

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