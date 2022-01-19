from typing import Sequence, Tuple
from workload import Task, TaskGraph


class BaseScheduler(object):

    def schedule(
            self,
            sim_time: float,
            released_tasks: Sequence[Task],
            task_graph: TaskGraph,
            worker_pools: 'Sequence[WorkerPool]',  # noqa: F821
    ) -> (float, Sequence[Tuple[Task, str]]):
        """ Abstract method to be implemented by derived classes to allow the
        scheduling of tasks.

        Args:
            sim_time (`float`): The time at which the scheduler is invoked.
            released_tasks (`Sequence[Task]`): The set of tasks released at
                the invocation of the current run of the scheduler.
            task_graph (`TaskGraph`): The state of the TaskGraph at this
                invocation including the future set of VIRTUAL tasks.
            worker_pools (`Sequence[WorkerPool]`): The set of worker pools to
                do the placement across.

        Returns:
            (scheduler_runtime, task_placement) where `scheduler_runtime` is a
            `float` depicting the runtime of the scheduler, and
            `task_placement` is a sequence of tuples depicting the
            (Task, ID of the Worker Pool where the task should be placed).
        """
        raise NotImplementedError("The `schedule()` method has not been "
                                  "implemented.")
