import os
import random
import time
from collections import defaultdict
from copy import copy, deepcopy
from enum import Enum
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Task, Workload
from workload.tasks import TaskGraph


class BranchPredictionScheduler(BaseScheduler):
    """Implements a branch prediction scheduling algorithm for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    class Policy(Enum):
        """Represents the different prediction policies used by the Scheduler."""

        WORST_CASE = 1  # Always predict the branch that will not be taken.
        BEST_CASE = 2  # Always predict the branch that will be taken.
        RANDOM = 3  # Randomly choose a branch to be taken.

    def __init__(
        self,
        policy: "Policy" = Policy.RANDOM,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        super(BranchPredictionScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )
        self._policy = policy

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Tuple[EventTime, Sequence[Tuple[Task, str, EventTime]]]:
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            sim_time,
            EventTime.zero(),
            self.preemptive,
            worker_pools,
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
                f"[{sim_time}] The state of {worker_pool} is:{os.linesep}"
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
            f"[{sim_time}] The order of the tasks is {ordered_task_names}."
        )

        # Run the scheduling loop.
        placements = []
        for task, _ in ordered_tasks:
            self._logger.debug(
                f"[{sim_time}] {self.__class__.__name__} trying to schedule {task} "
                f"with the resource requirements {task.resource_requirements}."
            )
            is_task_placed = False
            for worker_pool in schedulable_worker_pools.worker_pools:
                if worker_pool.can_accomodate_task(task):
                    worker_pool.place_task(task)
                    is_task_placed = True
                    placements.append((task, worker_pool.id, sim_time))
                    self._logger.debug(
                        f"[{sim_time}] Placed {task} on Worker Pool ({worker_pool.id})"
                        f" to be started at {sim_time}."
                    )
                    break

            if is_task_placed:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    self._logger.debug(
                        f"[{sim_time}] The state of {worker_pool} is:{os.linesep}"
                        f"{os.linesep.join(worker_pool.get_utilization())}"
                    )
            else:
                self._logger.debug(
                    f"[{sim_time}] Failed to place {task} because no worker pool "
                    f"could accomodate the resource requirements."
                )
                placements.append((task, None, None))

        end_time = time.time()
        if self.runtime == EventTime(time=-1, unit=EventTime.Unit.US):
            return (
                EventTime(int((end_time - start_time) * 1e6), EventTime.Unit.US),
                placements,
            )
        else:
            return self.runtime, placements

    def compute_slack(
        self, sim_time: EventTime, task_graph: TaskGraph, task_graph_name: str
    ) -> EventTime:
        remaining_time = self.compute_remaining_time(task_graph)
        expected_completion_time = sim_time + remaining_time
        self._logger.info(
            f"[{sim_time}] The deadline of the TaskGraph {task_graph_name} is "
            f"{task_graph.deadline}, and the remaining time is {remaining_time}. "
            f"The graph is expected to complete by {expected_completion_time}."
        )
        return task_graph.deadline - expected_completion_time

    def compute_remaining_time(self, task_graph: TaskGraph) -> EventTime:
        """Computes the slack of the given `TaskGraph` at the `sim_time` using
        the prediction policy specified in the Scheduler.

        Args:
            sim_time: The time at which to compute the slack.
            task_graph: The graph for which to compute the slack.

        Returns:
            The slack as per the specified branch prediction policy.
        """
        remaining_time = defaultdict(lambda: EventTime.zero())
        # Add the remaining time for the sources.
        for source_task in task_graph.get_source_tasks():
            remaining_time[source_task] = source_task.remaining_time

        # Iterate over the other nodes and choose the remaining
        # time according to the policy.
        for task in task_graph.topological_sort():
            # The branch associated with this task was not executed, skipping.
            if task not in remaining_time:
                continue

            if task.conditional:
                children_tasks = task_graph.get_children(task)
                # If the task is conditional, propagate the remaining time
                # to the children according to the policy in the scheduler.
                if self.policy == self.Policy.WORST_CASE:
                    # Choose the branch that has the lowest probability.
                    child_to_release = children_tasks[0]
                    for child in children_tasks[1:]:
                        if child.probability < child_to_release.probability:
                            child_to_release = child
                elif self.policy == self.Policy.BEST_CASE:
                    # Choose the branch that has the highest probability.
                    child_to_release = children_tasks[0]
                    for child in children_tasks[1:]:
                        if child.probability > child_to_release.probability:
                            child_to_release = child
                elif self.policy == self.Policy.RANDOM:
                    # Choose a branch randomly.
                    child_to_release = random.choice(children_tasks)
                else:
                    raise NotImplementedError(
                        f"The policy {self.policy} is not implemented yet."
                    )

                # Propagate the remaining time to the child.
                remaining_time[child_to_release] = (
                    remaining_time[task] + child_to_release.remaining_time
                )
            else:
                # If the task is not conditional, propogate the remaining time
                # to all the children.
                for child_task in task_graph.get_children(task):
                    if (
                        remaining_time[child_task]
                        <= remaining_time[task] + child_task.remaining_time
                    ):
                        remaining_time[child_task] = (
                            remaining_time[task] + child_task.remaining_time
                        )

        # Find the maximum remaining time across all the sink nodes.
        return max([remaining_time[sink] for sink in task_graph.get_sink_tasks()])

    @property
    def policy(self):
        return self._policy
