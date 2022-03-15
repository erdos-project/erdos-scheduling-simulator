from typing import Optional, Sequence, Tuple
from workload import Task, TaskGraph

import absl  # noqa: F401


class BaseScheduler(object):

    def schedule(
        self,
        sim_time: float,
        released_tasks: Sequence[Task],
        task_graph: TaskGraph,
        worker_pools: 'WorkerPools',  # noqa: F821
        _flags: Optional['absl.flags'] = None
    ) -> (float, Sequence[Tuple[Task, str]]):
        """ Abstract method to be implemented by derived classes to allow the
        scheduling of tasks.

        Args:
            sim_time (`float`): The time at which the scheduler is invoked.
            released_tasks (`Sequence[Task]`): The set of tasks released at
                the invocation of the current run of the scheduler.
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
        raise NotImplementedError("The `schedule()` method has not been "
                                  "implemented.")

    def log(self):
        raise NotImplementedError(
            "The `log()` method has not been implemented.")

    def _verify_schedule(self, tasks, dependency_matrix, start_times,
                         placements):
        # Check if each task's start time is greater than its release time.
        assert all([
            start_time >= task.release_time
            for start_time, task in zip(start_times, tasks)
        ]), "not_valid_release_times"

        # Check if all tasks finished before the deadline.
        assert all([(task.deadline >= start_time + task.runtime)
                    for task, start_time in zip(tasks, start_times)
                    ]), "doesn't finish before deadline"

        # Check if the task dependencies were satisfied.
        for i, row_i in enumerate(dependency_matrix):
            for j, col_j in enumerate(row_i):
                if i != j and col_j:
                    assert start_times[i] + tasks[i].runtime <= start_times[
                        j], f"not_valid_dependency{i}->{j}"

        # TODO: Check if resource requirements wever satisfied.
        for task, placement in zip(tasks, placements):
            pass

        # Check if tasks overlapped on a resource.
        placed_tasks = [(placement, start_time, start_time + task.runtime)
                        for placement, start_time, task in zip(
                            placements, start_times, tasks)]
        placed_tasks.sort()
        for t1, t2 in zip(placed_tasks, placed_tasks[1:]):
            if t1[0] == t2[0]:
                assert t1[2] <= t2[1], f"overlapping_tasks_on_{t1[0]}"
