import random
from collections import defaultdict
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
        pass

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
