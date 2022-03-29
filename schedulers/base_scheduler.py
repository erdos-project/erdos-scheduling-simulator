import heapq
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from workers import WorkerPools
from workload import Task, TaskGraph


class BaseScheduler(object):
    def schedule(
        self,
        sim_time: int,
        task_graph: TaskGraph,
        worker_pools: "WorkerPools",  # noqa: F821
        _flags: Optional["absl.flags"] = None,
    ) -> (int, Sequence[Tuple[Task, str]]):
        """Abstract method to be implemented by derived classes to allow the
        scheduling of tasks.

        Args:
            sim_time (`int`): The time in us at which the scheduler is invoked.
            task_graph (`TaskGraph`): The state of the TaskGraph at this
                invocation including the future set of VIRTUAL tasks.
            worker_pools (`WorkerPools`): The set of worker pools to
                do the placement across.

        Returns:
            (scheduler_runtime, task_placement) where `scheduler_runtime` is a `int`
            depicting the runtime of the scheduler (in us), and `task_placement` is
            a sequence of tuples depicting the (Task, ID of the Worker Pool where
            the task should be placed, Start time of the task (in us)).
        """
        raise NotImplementedError(
            "The `schedule()` method has not been " "implemented."
        )

    def log(self):
        raise NotImplementedError("The `log()` method has not been implemented.")

    @property
    def preemptive(self):
        raise NotImplementedError("The `preemptive` property has not been implemented.")

    @property
    def scheduling_horizon(self):
        raise NotImplementedError(
            "The `scheduling_horizon` property has not been implemented."
        )

    def _verify_schedule(
        self, worker_pools: WorkerPools, task_graph: TaskGraph, placements
    ):
        # Check if each task's start time is greater than its release time.
        assert all(
            [
                start_time >= task.release_time
                for task, placement, start_time in placements
            ]
        ), "not_valid_release_times"

        # Check if all tasks finished before the deadline.
        assert all(
            [
                (task.deadline >= start_time + task.runtime)
                for task, placement, start_time in placements
            ]
        ), "doesn't finish before deadline"

        # Check if task dependencies are satisfied.
        start_times = {task.id: start_time for task, _, start_time in placements}
        for task, placement, start_time in placements:
            children = task_graph.get_children(task)
            for child_task in children:
                if child_task.id in start_times:
                    assert (
                        start_time + task.remaining_time <= start_times[child_task.id]
                    ), f"task dependency not valid{task.id}->{child_task.id}"

        # Check if resource requirements are satisfied.
        placed_tasks = []
        for task, placement, start_time in placements:
            if placement is not None:
                placed_tasks.append((start_time, placement, task))
        placed_tasks.sort(key=lambda e: e[0])
        wps = deepcopy(worker_pools)
        id_to_wp = {wp.id: wp for wp in wps._wps}
        # A heap storing the task in order of their completion time.
        task_completion = []
        for start_time, wp_id, task in placed_tasks:
            # Remove task that finished before start_time.
            while len(task_completion) > 0 and start_time >= task_completion[0][0]:
                (end_time, wp_id, task) = heapq.heappop(task_completion)
                id_to_wp[wp_id].remove_task(task)
            wp = id_to_wp[wp_id]
            assert wp.can_accomodate_task(
                task
            ), f"WorkerPool {wp.id} doesn't have resources for {task.id}"
            wp.place_task(task)
            heapq.heappush(task_completion, (start_time + task.runtime, wp_id, task))
