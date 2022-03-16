from typing import Optional, Sequence, Tuple
from workload import Task, TaskGraph

import absl  # noqa: F401


class BaseScheduler(object):

    def schedule(
        self,
        sim_time: float,
        task_graph: TaskGraph,
        worker_pools: 'WorkerPools',  # noqa: F821
        _flags: Optional['absl.flags'] = None
    ) -> (float, Sequence[Tuple[Task, str]]):
        """ Abstract method to be implemented by derived classes to allow the
        scheduling of tasks.

        Args:
            sim_time (`float`): The time at which the scheduler is invoked.
            task_graph (`TaskGraph`): The state of the TaskGraph at this
                invocation including the future set of VIRTUAL tasks.
            worker_pools (`WorkerPools`): The set of worker pools to
                do the placement across.

        Returns:
            (scheduler_runtime, task_placement) where `scheduler_runtime` is a
            `float` depicting the runtime of the scheduler, and
            `task_placement` is a sequence of tuples depicting the
            (Task, ID of the Worker Pool where the task should be placed).
        """
        # TODO: We should probably include the start time in the return as
        # a scheduler may program a task to start at some time in the future
        # (probably before the next scheduler run completes).
        raise NotImplementedError("The `schedule()` method has not been "
                                  "implemented.")

    def log(self):
        raise NotImplementedError(
            "The `log()` method has not been implemented.")

    def _verify_schedule(self, task_graph, placements, start_times):
        # Check if each task's start time is greater than its release time.
        assert all([
            start_times[task.id] >= task.release_time
            for task, placement in placements
        ]), "not_valid_release_times"

        # Check if all tasks finished before the deadline.
        assert all([(task.deadline >= start_times[task.id] + task.runtime)
                    for task, placement in placements
                    ]), "doesn't finish before deadline"

        for task, placement in placements:
            children = task_graph.get_children(task)
            for child_task in children:
                assert (
                    start_times[task.id] + task.remaining_time <=
                    start_times[child_task.id]
                ), f"task dependency not valid{task.id}->{child_task.id}"

        # TODO: Check if resource requirements wever satisfied.
        for task, placement in placements:
            pass

        # Check if tasks overlapped on a resource.
        placed_tasks = [(placement, start_times[task.id],
                         start_times[task.id] + task.runtime)
                        for task, placement in placements]
        placed_tasks.sort()
        for t1, t2 in zip(placed_tasks, placed_tasks[1:]):
            if t1[0] == t2[0]:
                assert t1[2] <= t2[1], f"overlapping_tasks_on_{t1[0]}"
