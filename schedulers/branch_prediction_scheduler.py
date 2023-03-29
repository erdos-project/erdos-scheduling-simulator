import os
import time
from copy import copy, deepcopy
from typing import Optional

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import BranchPredictionPolicy, Placement, Placements, TaskGraph, Workload


class BranchPredictionScheduler(BaseScheduler):
    """Implements a branch prediction scheduling algorithm for the Simulator.

    Args:
        policy (`Policy`): The policy to choose the branch that going to be executed.
        preemptive (`bool`): If `True`, the scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        policy: BranchPredictionPolicy = BranchPredictionPolicy.RANDOM,
        branch_prediction_accuracy: float = 0.50,
        release_taskgraphs: bool = False,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        super(BranchPredictionScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            policy=policy,
            branch_prediction_accuracy=branch_prediction_accuracy,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            sim_time,
            EventTime.zero(),
            self.preemptive,
            worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
            release_taskgraphs=self.release_taskgraphs,
        )
        tasks_to_be_scheduled = [
            (
                task,
                self.compute_slack(
                    sim_time, workload.get_task_graph(task.task_graph), task.task_graph
                ),
            )
            for task in tasks_to_be_scheduled
        ]

        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = copy(worker_pools)

        for worker_pool in schedulable_worker_pools.worker_pools:
            self._logger.debug(
                f"[{sim_time.time}] The state of {worker_pool} is:{os.linesep}"
                f"{os.linesep.join(worker_pool.get_utilization())}"
            )

        # Sort the tasks according to their slack, and place them on the
        # worker pools.
        start_time = time.time()
        ordered_tasks = list(sorted(tasks_to_be_scheduled, key=lambda item: item[1]))

        ordered_task_names = [
            f"{task.unique_name}({slack})" for task, slack in ordered_tasks
        ]
        self._logger.info(
            f"[{sim_time.time}] The order of the tasks is {ordered_task_names}."
        )

        # Run the scheduling loop.
        placements = []
        for task, _ in ordered_tasks:
            self._logger.debug(
                f"[{sim_time.time}] {self.__class__.__name__} trying to schedule "
                f"{task} with the resource requirements {task.resource_requirements}."
            )
            is_task_placed = False
            for execution_strategy in task.available_execution_strategies:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    if worker_pool.can_accomodate_strategy(execution_strategy):
                        worker_pool.place_task(task, execution_strategy)
                        is_task_placed = True
                        placements.append(
                            Placement(
                                task=task,
                                placement_time=sim_time,
                                worker_pool_id=worker_pool.id,
                                execution_strategy=execution_strategy,
                            )
                        )
                        self._logger.debug(
                            f"[{sim_time.time}] Placed {task} on WorkerPool "
                            f"({worker_pool.id}) to be started at {sim_time}, and "
                            f"executed using the strategy {execution_strategy}."
                        )
                        break

            if is_task_placed:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    self._logger.debug(
                        f"[{sim_time.time}] The state of {worker_pool} is:{os.linesep}"
                        f"{os.linesep.join(worker_pool.get_utilization())}"
                    )
            else:
                self._logger.debug(
                    f"[{sim_time.time}] Failed to place {task} because no worker pool "
                    f"could accomodate the resource requirements."
                )
                placements.append(Placement(task=task))

        end_time = time.time()

        # Compute and return the Placements object.
        scheduler_runtime = EventTime(
            int((end_time - start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime
            if self.runtime == EventTime(-1, EventTime.Unit.US)
            else self.runtime
        )
        return Placements(
            runtime=runtime,
            true_runtime=scheduler_runtime,
            placements=placements,
        )

    def compute_slack(
        self, sim_time: EventTime, task_graph: TaskGraph, task_graph_name: str
    ) -> EventTime:
        remaining_time = task_graph.get_remaining_time(self.policy)
        expected_completion_time = sim_time + remaining_time
        self._logger.info(
            f"[{sim_time.time}] The deadline of the TaskGraph {task_graph_name} is "
            f"{task_graph.deadline}, and the remaining time is {remaining_time}. "
            f"The graph is expected to complete by {expected_completion_time}."
        )
        return task_graph.deadline - expected_completion_time
