from enum import Enum
from functools import total_ordering
from typing import Optional, Sequence, Union

from utils import EventTime

from .strategy import ExecutionStrategy


class Placement(object):
    """A mapping of a particular Task / WorkProfile to its executing Worker determined
    by a Scheduler.

    The class provides the required information computed by a Scheduler that aids the
    simulator in placing the available tasks to the available resources. An instance of
    this class should be created using one of its static methods.

    Args:
        type (`PlacementType`): The type of the Placement (for loading / unloading work,
            or placing a task).
        computation (`Union[Task, WorkProfile]`): The computation that is represented
            by this Placement. The available type of the computation depends on the
            type of the Placement specified by its `type`.
        placement_time (`Optional[EventTime]`): The time at which the placement is
            supposed to be initiated. If None, the Task was unplaced.
        worker_pool_id (`Optional[str]`): The ID of the WorkerPool where the Task is
            to be placed. If None, the Task was unplaced.
        worker_id (`Optional[str]`): The ID of the Worker where the Task is to be
            placed. If None, the Worker is assigned by the second-level Scheduler used
            inside the `WorkerPool`.
        strategy (`ExecutionStrategy`): The strategy to be used for executing this
            Placement.
    """

    @total_ordering
    class PlacementType(Enum):
        """The different types of Placement actions available to a Scheduler."""

        # An action that informs the Simulator to evict the WorkProfile.
        EVICT_WORK_PROFILE = 1
        # An action that informs the Simulator to load the WorkProfile.
        LOAD_WORK_PROFILE = 2
        # An action that informs the Simulator to place a Task.
        PLACE_TASK = 3

        def __str__(self) -> str:
            if self == Placement.PlacementType.EVICT_WORK_PROFILE:
                return "EVICT_WORK_PROFILE"
            elif self == Placement.PlacementType.LOAD_WORK_PROFILE:
                return "LOAD_WORK_PROFILE"
            elif self == Placement.PlacementType.PLACE_TASK:
                return "PLACE_TASK"

        def __repr__(self) -> str:
            return str(self)

        def __lt__(self, other) -> bool:
            return self.value < other.value

        def __eq__(self, other) -> bool:
            return self.value == other.value

    def __init__(
        self,
        type: PlacementType,
        computation: Union["Task", "WorkProfile"],  # noqa: F821
        placement_time: Optional[EventTime] = None,
        worker_pool_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        strategy: ExecutionStrategy = None,
    ) -> None:
        self._placement_type = type
        self._computation = computation
        self._placement_time = placement_time
        self._worker_pool_id = worker_pool_id
        self._worker_id = worker_id
        self._strategy = strategy

    def is_placed(self) -> bool:
        """Check if the computation associated with this Placement was placed on a
        WorkerPool.

        The method is only available for `PLACE_TASK` placement type.

        Returns:
            `True` if the task was placed, `False` otherwise.

        Raises:
            `RuntimeError` if the type of the placement is not `PLACE_TASK`.
        """
        if self._placement_type != Placement.PlacementType.PLACE_TASK:
            raise RuntimeError(
                f"A Placement of type {str(self._placement_type)} cannot invoke "
                f"the `is_placed` method."
            )
        return self.worker_pool_id is not None

    @property
    def work_profile(self) -> "WorkProfile":  # noqa: F821
        """Returns the `WorkProfile` for which this placement was specified.

        This method is only available for `LOAD_WORK_PROFILE` and `EVICT_WORK_PROFILE`
        placement types.

        Raises:
            `RuntimeError` if the type of the placement is not `LOAD_WORK_PROFILE` or
            `EVICT_WORK_PROFILE`.
        """
        if self._placement_type not in (
            Placement.PlacementType.LOAD_WORK_PROFILE,
            Placement.PlacementType.EVICT_WORK_PROFILE,
        ):
            raise RuntimeError(
                f"A Placement of type {str(self._placement_type)} does not have a "
                f"WorkProfile object."
            )
        return self._computation

    @property
    def task(self) -> "Task":  # noqa: F821
        """Returns the `Task` for which this placement was specified.

        This method is only available for `PLACE_TASK` placement type.

        Raises:
            `RuntimeError` if the type of the placement is not `PLACE_TASK`.
        """
        if self._placement_type != Placement.PlacementType.PLACE_TASK:
            raise RuntimeError(
                f"A Placement of type {str(self._placement_type)} does not have a "
                f"Task object."
            )
        return self._computation

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
    def placement_type(self) -> "Placement.PlacementType":
        """Returns the Placement type of this instance of `Placement`."""
        return self._placement_type

    @property
    def id(self) -> str:
        """Returns the ID of the computation underlying this Placement."""
        return self._computation.id

    @property
    def name(self) -> str:
        """Returns the name of the computation underlying this Placement."""
        return self._computation.name

    @property
    def execution_strategy(self) -> Optional[ExecutionStrategy]:
        """Returns the strategy to be used by a `Task` for its execution.

        This method is only available for `PLACE_TASK` placement type.

        Raises:
            `RuntimeError` if the type of the placement is not `PLACE_TASK`.
        """
        if self._placement_type != Placement.PlacementType.PLACE_TASK:
            raise RuntimeError(
                f"A Placement of type {self._placement_type} does not have an "
                f"execution strategy."
            )
        return self._strategy

    @property
    def loading_strategy(self) -> Optional[ExecutionStrategy]:
        """Returns the strategy to be used by a `WorkProfile` for its loading.

        This method is only available for `LOAD_PROFILE` and `EVICT_PROFILE` placement
        type.

        Raises:
            `RuntimeError` if the type of the placement is not `LOAD_PROFILE` or
            `EVICT_PROFILE`.
        """
        if self._placement_type not in (
            Placement.PlacementType.LOAD_WORK_PROFILE,
            Placement.PlacementType.EVICT_WORK_PROFILE,
        ):
            raise RuntimeError(
                f"A Placement of type {self._placement_type} does not have a "
                f"loading strategy."
            )
        return self._strategy

    def __str__(self) -> str:
        if self._placement_type in (
            Placement.PlacementType.LOAD_WORK_PROFILE,
            Placement.PlacementType.EVICT_WORK_PROFILE,
        ):
            return (
                f"Placement(type={self._placement_type}, "
                f"work_profile={self.work_profile}, "
                f"time={self.placement_time}, worker_pool_id={self.worker_pool_id}, "
                f"worker_id={self.worker_id}, "
                f"loading_strategy={self.loading_strategy})"
            )
        elif self._placement_type == Placement.PlacementType.PLACE_TASK:
            return (
                f"Placement(type={self._placement_type}, task={self.task.unique_name}, "
                f"time={self.placement_time}, worker_pool_id={self.worker_pool_id}, "
                f"worker_id={self.worker_id}, "
                f"execution_strategy={self.execution_strategy})"
            )

    @staticmethod
    def create_task_placement(
        task: "Task",  # noqa: F821
        placement_time: Optional[EventTime] = None,
        worker_pool_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        execution_strategy: ExecutionStrategy = None,
    ) -> "Placement":
        return Placement(
            type=Placement.PlacementType.PLACE_TASK,
            computation=task,
            placement_time=placement_time,
            worker_pool_id=worker_pool_id,
            worker_id=worker_id,
            strategy=execution_strategy,
        )

    @staticmethod
    def create_load_profile_placement(
        work_profile: "WorkProfile",  # noqa: F821
        placement_time: EventTime,
        worker_pool_id: str,
        loading_strategy: ExecutionStrategy,
        worker_id: Optional[str] = None,
    ) -> "Placement":
        return Placement(
            type=Placement.PlacementType.LOAD_WORK_PROFILE,
            computation=work_profile,
            placement_time=placement_time,
            worker_pool_id=worker_pool_id,
            worker_id=worker_id,
            strategy=loading_strategy,
        )

    @staticmethod
    def create_evict_profile_placement(
        work_profile: "WorkProfile",  # noqa: F821
        placement_time: EventTime,
        worker_pool_id: str,
        worker_id: Optional[str] = None,
    ) -> "Placement":
        return Placement(
            type=Placement.PlacementType.EVICT_WORK_PROFILE,
            computation=work_profile,
            placement_time=placement_time,
            worker_pool_id=worker_pool_id,
            worker_id=worker_id,
            strategy=None,
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
        self._placements = {placement.id: placement for placement in placements}

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

    def get_placement(
        self, computation: Union["Task", "WorkProfile"]  # noqa: F821
    ) -> Optional[Placement]:  # noqa: F821
        """Retrieves the placement for the corresponding computation (if available).

        Args:
            computation (`Union[Task, WorkProfile]`): The computation for which the
                placement is to be retrieved.

        Returns:
            The `Placement` object if found, `None` otherwise.
        """
        return self._placements.get(computation.id)

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
            + ", ".join(placement.name for placement in self._placements.values())
            + "])"
        )

    def __repr__(self):
        return str(self)
