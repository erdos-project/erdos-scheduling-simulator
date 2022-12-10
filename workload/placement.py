from typing import Optional, Sequence

from utils import EventTime

from .tasks import Task


class Placement(object):
    """A mapping of a particular Task to its executing Worker determined by a Scheduler.

    The class provides the required information computed by a Scheduler that aids the
    simulator in placing the available tasks to the available resources.

    Args:
        task (`Task`): The task for which the placement is specified.
        worker_pool_id (`Optional[str]`): The ID of the WorkerPool where the Task is
            to be placed. If None, the Task was unplaced.
        placement_time (`Optional[EventTime]`): The time at which the placement is
            supposed to be initiated. If None, the Task was unplaced.
    """

    def __init__(
        self,
        task: Task,
        worker_pool_id: Optional[str] = None,
        placement_time: Optional[EventTime] = None,
    ) -> None:
        self._task = task
        self._worker_pool_id = worker_pool_id
        self._placement_time = placement_time

    def is_placed(self) -> bool:
        """Check if the task associated with this Placement was placed on a WorkerPool.

        Returns:
            `True` if the task was placed, `False` otherwise.
        """
        return self.worker_pool_id is not None

    @property
    def task(self) -> Task:
        """Returns the `Task` for which this placement was specified."""
        return self._task

    @property
    def worker_pool_id(self) -> Optional[str]:
        """Returns the ID of the `WorkerPool` where the Task is to be placed."""
        return self._worker_pool_id

    @property
    def placement_time(self) -> Optional[EventTime]:
        """Returns the time at which the `Task` is supposed to be placed
        on the `WorkerPool`."""
        return self._placement_time


class Placements(object):
    """A representation of the placements determined by a Scheduler.

    The class acts as a wrapper over the individual `Placement` objects for each of the
    Tasks asked to be placed by the Scheduler.

    Args:
        runtime (`EventTime`): The runtime of the Scheduler invocation.
        placements (`Sequence[Placement]`): The placements decided by the scheduler.
    """

    def __init__(
        self,
        runtime: EventTime,
        true_runtime: Optional[EventTime] = None,
        placements: Sequence[Placement] = [],
    ) -> None:
        self._runtime = runtime
        self._true_runtime = true_runtime
        self._placements = {placement.task.id: placement for placement in placements}

    def add_placement(
        self,
        task: Task,
        worker_pool_id: Optional[str] = None,
        placement_time: Optional[EventTime] = None,
    ) -> None:
        """Adds the given placement to the Placements decided by the Scheduler.

        Args:
            task (`Task`): The Task for which the Placement is to be added.
            worker_pool_id (`Optional[str]`): The ID of the WorkerPool where the task
                is to be placed. If None, the task was unplaced.
            placement_time (`Optional[EventTime]`): The time at which the task is
                supposed to be placed. If None, the task was unplaced.
        """
        self._placements[task.id] = Placement(task, worker_pool_id, placement_time)

    def get_placement(self, task: Task) -> Optional[Placement]:
        """Retrieves the placement for the corresponding task.

        Args:
            task (`Task`): The task for which the placement is to be retrieved.

        Returns:
            The `Placement` object if found, `None` otherwise.
        """
        return self._placements.get(task.id)

    @property
    def runtime(self) -> EventTime:
        """Returns the runtime of the Scheduler invocation."""
        return self._runtime

    @property
    def true_runtime(self) -> Optional[EventTime]:
        """Returns the actual runtime of the Scheduler invocation."""
        return self._true_runtime

    def __iter__(self):
        for item in self._placements.values():
            yield item

    def __len__(self):
        return len(self._placements)

    def __str__(self):
        return (
            "Placements(["
            + ", ".join(
                placement.task.unique_name for placement in self._placements.values()
            )
            + "])"
        )

    def __repr__(self):
        return str(self)
