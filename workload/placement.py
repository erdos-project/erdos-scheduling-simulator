from typing import Optional, Sequence

from utils import EventTime

from .strategy import ExecutionStrategy


class Placement(object):
    """A mapping of a particular Task to its executing Worker determined by a Scheduler.

    The class provides the required information computed by a Scheduler that aids the
    simulator in placing the available tasks to the available resources.

    Args:
        task (`Task`): The task for which the placement is specified.
        placement_time (`Optional[EventTime]`): The time at which the placement is
            supposed to be initiated. If None, the Task was unplaced.
        worker_pool_id (`Optional[str]`): The ID of the WorkerPool where the Task is
            to be placed. If None, the Task was unplaced.
        worker_id (`Optional[str]`): The ID of the Worker where the Task is to be
            placed. If None, the Worker is assigned by the second-level Scheduler used
            inside the `WorkerPool`.
        execution_strategy (`ExecutionStrategy`): The execution strategy to be
            observed by this placement.
    """

    def __init__(
        self,
        task: "Task",  # noqa: F821
        placement_time: Optional[EventTime] = None,
        worker_pool_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        execution_strategy: ExecutionStrategy = None,
    ) -> None:
        self._task = task
        self._placement_time = placement_time
        self._worker_pool_id = worker_pool_id
        self._worker_id = worker_id
        self._execution_strategy = execution_strategy

    def is_placed(self) -> bool:
        """Check if the task associated with this Placement was placed on a WorkerPool.

        Returns:
            `True` if the task was placed, `False` otherwise.
        """
        return self.worker_pool_id is not None

    @property
    def task(self) -> "Task":  # noqa: F821
        """Returns the `Task` for which this placement was specified."""
        return self._task

    @property
    def placement_time(self) -> Optional[EventTime]:
        """Returns the time at which the `Task` is supposed to be placed
        on the `WorkerPool`."""
        return self._placement_time

    @property
    def worker_pool_id(self) -> Optional[str]:
        """Returns the ID of the `WorkerPool` where the Task is to be placed."""
        return self._worker_pool_id

    @property
    def worker_id(self) -> Optional[str]:
        """Returns the ID of the `Worker` where the Task is to be placed."""
        return self._worker_id

    @property
    def execution_strategy(self) -> Optional[ExecutionStrategy]:
        """Returns the ExecutionStrategy to be used by the Task."""
        return self._execution_strategy

    def __str__(self) -> str:
        return (
            f"Placement(task={self.task.unique_name}, time={self.placement_time}, "
            f"worker_pool_id={self.worker_pool_id}, worker_id={self.worker_id}, "
            f"execution_strategy={self.execution_strategy})"
        )


class Placements(object):
    """A representation of the placements determined by a Scheduler.

    The class acts as a wrapper over the individual `Placement` objects for each of the
    Tasks asked to be placed by the Scheduler.

    Args:
        runtime (`EventTime`): The runtime of the Scheduler invocation as applied on
            the Simulator.
        true_runtime (`EventTime`): The actual runtime of the Scheduler invocation.
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
        task: "Task",  # noqa: F821
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

    def get_placement(self, task: "Task") -> Optional[Placement]:  # noqa: F821
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
