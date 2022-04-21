import heapq
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from workers import WorkerPools
from workload import Task, TaskGraph


class BaseScheduler(object):
    """Base class that the schedulers must implement.

    Args:
        preemptive (`bool`): If `True`, the scheduler can preempt the tasks that
            are currently running.
        runtime (`int`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
        scheduling_horizon (`int`): The scheduler will try to place tasks that are
            within the scheduling horizon (in us) using estimated task release times.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        scheduling_horizon: int = 0,
        enforce_deadlines: bool = False,
    ):
        self._preemptive = preemptive
        self._runtime = runtime
        self._scheduling_horizon = scheduling_horizon
        self._enforce_deadlines = enforce_deadlines

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
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime

    @property
    def enforce_deadlines(self):
        return self._enforce_deadlines

    @property
    def scheduling_horizon(self):
        return self._scheduling_horizon

    def _verify_schedule(
        self, worker_pools: WorkerPools, task_graph: TaskGraph, placements
    ):
        placed_tasks = [
            (start_time, placement, task)
            for task, placement, start_time in placements
            if placement is not None
        ]
        for task, placement, start_time in placed_tasks:
            # Ensure that the task didn't start before it was released.
            assert (
                start_time >= task.release_time
            ), f"Task {task} start time {start_time} is less than the release time"
            if self._enforce_deadlines:
                # Check if the task finished before the deadline.
                assert (
                    task.deadline >= start_time + task.runtime
                ), f"Task {task} did not finish before its deadline"

        # Check if task dependencies are satisfied.
        start_times = {task.id: start_time for task, _, start_time in placed_tasks}
        for task, placement, start_time in placed_tasks:
            children = task_graph.get_children(task)
            # Check that the children of the placed task weren't scheduled to
            # start before the task completes.
            for child_task in children:
                if child_task.id in start_times:
                    assert (
                        start_time + task.remaining_time <= start_times[child_task.id]
                    ), f"Task dependency not valid{task.id}->{child_task.id}"

        # Check if resource requirements are satisfied.
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
            wp.place_task(task, dry_run=True)
            heapq.heappush(task_completion, (start_time + task.runtime, wp_id, task))
