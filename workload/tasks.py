import logging
import random
import sys
import uuid
from collections import defaultdict, deque
from enum import Enum
from functools import cached_property, total_ordering
from typing import Mapping, Optional, Sequence, Tuple, Union

from utils import EventTime, setup_logging

from . import BranchPredictionPolicy
from .graph import Graph
from .placement import Placement
from .profile import WorkProfile
from .strategy import ExecutionStrategies, ExecutionStrategy


@total_ordering
class TaskState(Enum):
    """Represents the different states that a Task can potentially be in."""

    VIRTUAL = 1  # The Task is expected to be generated in the future.
    RELEASED = 2  # The Task has been released, and is waiting to be scheduled.
    SCHEDULED = 3  # The Task has been scheduled, and is waiting for execution.
    RUNNING = 4  # The Task has begun execution, and is currently running.
    PREEMPTED = 5  # The Task had begun execution but is currently paused.
    EVICTED = 6  # The Task has been evicted before completing.
    COMPLETED = 7  # The Task has successfully completed.
    CANCELLED = 8  # The Task has been cancelled, and will not be scheduled.

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __eq__(self, other) -> bool:
        return self.value == other.value


# Only PREEMPTED, SCHEDULED and VIRTUAL tasks can be released.
RELEASABLE_TASK_STATES = (TaskState.VIRTUAL, TaskState.SCHEDULED, TaskState.PREEMPTED)


@total_ordering
class Task(object):
    """A `Task` represents the invocation of a computation of a particular
    Operator in an ERDOS AV pipeline.

    A `Task` is a runtime entity and contains an estimated runtime, a deadline
    and a request for a set of resources. A set of `Task`s form a `TaskGraph`,
    with their dependencies being governed by timestamps or the structure of
    the computation defined in a `JobGraph`.

    Args:
        name (`str`): The name of the computation (typically the callback of
            the ERDOS operator.
        task_graph (`str`): The name of the TaskGraph that this Task belongs to.
        job (`Job`): The job that created this particular task.
        deadline (`EventTime`): The absolute deadline by which the task should complete.
        profile (`WorkProfile`): A profile of the computation that the Task is supposed
            to execute (including the execution strategies for different resources).
        release_time (`EventTime`): The time at which the task was released by the
            job (in us).
        state (`TaskState`): Defines the state of the task.
        timestamp (`int`): The timestamp for the Task (single dimension).
        start_time (`EventTime`): The time (in us) at which the task was started
            (only available if state != TaskState.RELEASED, -1 otherwise)
        remaining_time (`EventTime`): The time (in us) remaining to finish the
            completion of the task.
        completion_time (`EventTime`): The time (in us) at which the task completed
            / was preempted (only available if state is either
            EVICTED / COMPLETED, -1 otherwise)
        probability (`float`): The probability with which this Task will be executed.
            If `None`, the probability is retrieved from the creating Job.
        _logger(`Optional[logging.Logger]`): The logger to use to log the
            results of the execution.
    """

    class Preemption:
        def __init__(self, preemption_time, old_worker_pool):
            self.preemption_time = preemption_time
            self.old_worker_pool = old_worker_pool
            self.restart_time = None
            self.new_worker_pool = None

    def __init__(
        self,
        name: str,
        task_graph: str,
        job: "Job",  # noqa: F821
        deadline: EventTime,
        profile: Optional[WorkProfile] = None,
        timestamp: Optional[int] = None,
        release_time: EventTime = EventTime.invalid(),
        start_time: EventTime = EventTime.invalid(),
        completion_time: EventTime = EventTime.invalid(),
        probability: Optional[float] = None,
        _logger: Optional[logging.Logger] = None,
    ):
        if (
            completion_time != EventTime.invalid()
            and profile
            and len(profile.execution_strategies) != 1
        ):
            raise RuntimeError(
                "Execution of TaskGraphs with multiple strategies and a "
                "fixed completion time is not supported."
            )
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = setup_logging(name=f"{name}_{timestamp}")

        self._name = name
        self._task_graph = task_graph
        self._creating_job = job
        self._probability = (
            self._creating_job.probability if probability is None else probability
        )
        self._profile = profile if profile else job.profile
        self._deadline = deadline
        self._timestamp = timestamp
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._hash = hash(self._id)

        # The timestamps maintained for each state of the task.
        # (VIRTUAL -> RELEASED)
        self._intended_release_time = release_time
        self._release_time = release_time
        # (RELEASED -> SCHEDULED)
        self._scheduling_time = None
        self._scheduler_placement = None
        # (SCHEDULED -> RUNNING)
        self._start_time = start_time
        # (SCHEDULED -> CANCELLED)
        self._cancellation_time = None
        # (RUNNING -> EVICTED / COMPLETED)
        self._completion_time = completion_time
        # (RUNNING -> PREEMPTED)
        self._preemptions = []

        # The data required for managing the execution of a particular task.
        self._remaining_time = None
        self._last_step_time = -1  # Time when this task was stepped through.
        self._state = TaskState.VIRTUAL
        self._pre_scheduling_state = TaskState.VIRTUAL
        # ID of the worker pool on which the task is running.
        self._worker_pool_id = None

    def release(self, time: Optional[EventTime] = None):
        """Release the task and transition away from the virtual state.

        Args:
            time (`Optional[EventTime]`): The simulation time (in us) at which to
                release the task. If None, should be specified at task construction.
        """
        if time and type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if time is None and self._release_time.is_invalid():
            raise ValueError(
                "Release time should be specified either while "
                "creating the Task or when releasing it."
            )
        if self.state not in RELEASABLE_TASK_STATES:
            raise ValueError(
                f"The task {self.unique_name} cannot be released from "
                f"state {self.state}."
            )
        if time is not None:
            self._logger.debug(
                f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} "
                f"to {TaskState.RELEASED}."
            )
            self._release_time = time
            if self._release_time > self._deadline:
                self._logger.warning(
                    f"[{time.to(EventTime.Unit.US).time}] Task {self} was released "
                    f"after its deadline {self._deadline} and the intended release "
                    f"time was {self.intended_release_time}."
                )
        if self._state < TaskState.RELEASED:
            if time:
                self._logger.debug(
                    f"[{time.to(EventTime.Unit.US).time}] Released task "
                    f"{self.unique_name} from state {self._state}."
                )
            self._state = TaskState.RELEASED
            self._pre_scheduling_state = TaskState.RELEASED

    def schedule(
        self,
        time: EventTime,
        placement: Placement,
    ):
        """Schedules the execution of the task at the given simulator time.

        Args:
            time (`EventTime`): The simulation time at which the task was
                scheduled.
            placement (`Placement`): The Placement information returned by the
                Scheduler.

        Raises:
            `ValueError` if Task is not in `VIRTUAL`/`RELEASED`/`PREEMPTED`
            state yet.
        """
        if self.state not in (
            TaskState.VIRTUAL,
            TaskState.RELEASED,
            TaskState.PREEMPTED,
            # Allow previously scheduled tasks to be rescheduled for a later
            # time or to a different WorkerPool.
            TaskState.SCHEDULED,
        ):
            raise ValueError(
                f"Only VIRTUAL/RELEASED/PREEMPTED/SCHEDULED tasks can be scheduled. "
                f"Task is in state {self.state}."
            )
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} to "
            f"{TaskState.SCHEDULED} to be started at {placement.placement_time} "
            f"on WorkerPool({placement.worker_pool_id})."
        )
        self._state = TaskState.SCHEDULED
        self._scheduling_time = time
        self._scheduler_placement = placement
        self._start_time = placement.placement_time
        self._worker_pool_id = placement.worker_pool_id
        self.update_remaining_time(placement.execution_strategy.runtime)

    def unschedule(
        self,
        time: EventTime,
    ) -> None:
        """Unschedules the execution of the task at the given simulator time.

        Args:
            time (`EventTime`): The simulation time at which the task was
                unscheduled.

        Raises:
            `ValueError` if Task is not in `SCHEDULED` state yet.
        """
        if self.state != TaskState.SCHEDULED:
            raise ValueError(
                f"Task must be in SCHEDULED state, currently in {self.state}."
            )
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} to "
            f"{self._pre_scheduling_state} from {TaskState.SCHEDULED}."
        )
        self._state = self._pre_scheduling_state
        self._scheduling_time = None
        self._scheduler_placement = None
        self._worker_pool_id = None

    def start(
        self,
        time: Optional[EventTime] = None,
        variance: Optional[int] = 0,
    ):
        """Begins the execution of the task at the given simulator time.

        Args:
            time (`Optional[EventTime]`): The simulation time (in us) at which to
                begin the task. If None, should be specified at task construction.
            variance (`Optional[int]`): The percentage variation to add to
                the runtime of the task.

        Raises:
            `ValueError` if Task is not in `SCHEDULED` state yet.
        """
        if self.state != TaskState.SCHEDULED:
            raise ValueError(
                f"Only SCHEDULED tasks can be started. Task is in state {self.state}."
            )
        if time is not None and type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if time is None and self._start_time == EventTime.invalid():
            raise ValueError(
                "Start time should be specified either while "
                "creating the Task or when starting it."
            )

        remaining_time = self._remaining_time.fuzz((0, variance))
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} to "
            f"{TaskState.RUNNING} with the remaining time {remaining_time}."
        )
        self._start_time = time if time is not None else self._start_time
        assert (
            self._start_time >= self._release_time
        ), f"Task {self.id} start time must be greater than release time"
        self._last_step_time = time
        self._state = TaskState.RUNNING
        self.update_remaining_time(remaining_time)

    def step(
        self,
        current_time: EventTime,
        step_size: EventTime = EventTime(1, EventTime.Unit.US),
    ) -> bool:
        """Steps the task for the given `step_size` (default 1 time step).

        Args:
            current_time (`EventTime`): The current time of the simulator (in us).
            step_size (`EventTime`): The amount of time (in us) for which to step
                the task.

        Returns:
            `True` if the task has finished execution, `False` otherwise.
        """
        if type(current_time) != EventTime:
            raise ValueError(
                f"Invalid type received for current_time: {type(current_time)}"
            )
        if type(step_size) != EventTime:
            raise ValueError(f"Invalid type received for step_size: {type(step_size)}")
        if (
            self.state != TaskState.RUNNING
            or self.start_time > current_time + step_size
        ):
            # We cannot step a Task that's not supposed to be running.
            self._logger.warning(
                f"[{current_time.to(EventTime.Unit.US).time}] Cannot step {self} with "
                f"start time {self.start_time} at time {current_time} since it's "
                f"either not RUNNING or isn't supposed to start yet."
            )
            return False

        if self._remaining_time == EventTime.zero():
            self._logger.info(
                "[%d] Skipping the stepping of %s since it "
                "has zero remaining time left.",
                current_time.to(EventTime.Unit.US).time,
                self,
            )
            return False

        # Task can be run, step through the task's execution.
        execution_time = current_time + step_size - self._last_step_time
        if self._remaining_time - execution_time <= EventTime.zero():
            self._last_step_time = current_time + self._remaining_time
            self._remaining_time = EventTime.zero()
            return True
        else:
            self._last_step_time = current_time + step_size
            self._remaining_time -= execution_time
            self._logger.debug(
                f"[{current_time.to(EventTime.Unit.US).time}] Stepped {self} for "
                f"{step_size} steps. Remaining execution time: {self._remaining_time}."
            )
            return False

    def preempt(self, time: EventTime):
        """Preempts the task at the given simulation time.

        Args:
            time (`EventTime`): The simulation time (in us) at which to preempt the
                task.

        Raises:
            `ValueError` if task is not RUNNING.
        """
        if type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if self.state != TaskState.RUNNING:
            raise ValueError(f"Task {self.id} is not RUNNING right now.")
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} to "
            f"{TaskState.PREEMPTED}."
        )
        self._preemptions.append(
            Task.Preemption(
                preemption_time=time,
                old_worker_pool=self._worker_pool_id,
            )
        )
        self._state = TaskState.PREEMPTED
        self._worker_pool_id = None

    def resume(self, time: EventTime, worker_pool_id: Optional[str] = None):
        """Continues the execution of the task at the given simulation time.

        If the `worker_pool_id` passed is `None`, it is assumed that the task
        will be restarted at the old worker pool.

        Args:
            time (`EventTime`): The simulation time (in us) at which to restart the
                task.
            worker_pool_id (`Optional[str]`): The ID of the WorkerPool that
                the task will be resumed on.

        Raises:
            `ValueError` if task is not PREEMPTED.
        """
        if type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if self.state != TaskState.PREEMPTED:
            raise ValueError(f"Task {self.id} is not PREEMPTED right now.")
        new_worker_pool = (
            worker_pool_id if worker_pool_id else self.last_preemption.old_worker_pool
        )
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Transitioning {self} which was "
            f"PREEMPTED at {self.preemption_time} to {TaskState.RUNNING} on "
            f"WorkerPool ({new_worker_pool})."
        )
        self.last_preemption.restart_time = time
        self.last_preemption.new_worker_pool = new_worker_pool
        self._last_step_time = time
        self._state = TaskState.RUNNING
        self._worker_pool_id = new_worker_pool

    def finish(self, time: Optional[EventTime] = None):
        """Completes the execution of the task at the given simulation time.

        If the remaining time is not 0, the task is considered to be preempted.

        Args:
            time (`EventTime`): The simulation time (in us) at which the task was
                finished. If `None`, the last step time is used.
        """
        if self.state not in [TaskState.RUNNING, TaskState.PREEMPTED]:
            raise ValueError(f"Task {self.id} is not RUNNING or PREEMPTED right now.")
        self._completion_time = time if time is not None else self._last_step_time
        if self._remaining_time == EventTime.zero():
            self._state = TaskState.COMPLETED
        else:
            self._state = TaskState.EVICTED

        self._worker_pool_id = None
        self._logger.debug(
            f"[{self._completion_time.to(EventTime.Unit.US).time}] Finished "
            f"execution of {self}."
        )

    def cancel(self, time: EventTime) -> None:
        """Cancels the pending execution of the Task.

        This method can be used to cancel a VIRTUAL, RELEASED or SCHEDULED task.
        However, a currently RUNNING or PREEMPTED task must not be cancelled.

        Args:
            time: The time at which the Task was cancelled.
        """
        if type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if self.state not in (
            TaskState.VIRTUAL,
            TaskState.RELEASED,
            TaskState.SCHEDULED,
        ):
            raise ValueError(
                f"Task {self.unique_name} is in a non-cancellable state: {self.state}."
            )
        self._cancellation_time = time
        self.update_probability(0.0)
        self.update_remaining_time(EventTime.zero())
        self._state = TaskState.CANCELLED
        self._logger.debug(
            f"[{time.to(EventTime.Unit.US).time}] Cancelled execution of {self}."
        )

    def update_remaining_time(self, time: EventTime):
        """Updates the remaining time of the task to simulate any runtime
        variabilities.

        Args:
            time (`EventTime`): The new remaining time (in us) to update the task
                with.

        Raises:
            `ValueError` if the task is COMPLETED / EVICTED, or time < 0.
        """
        if type(time) != EventTime:
            raise ValueError(f"Invalid type received for time: {type(time)}")
        if self.is_complete():
            raise ValueError(
                f"The remaining time of COMPLETED/EVICTED "
                f"task {self.id} cannot be updated."
            )
        if time < EventTime.zero():
            raise ValueError("Trying to set a negative value for remaining time.")
        self._remaining_time = time

    def update_deadline(self, new_deadline: EventTime):
        """Updates the absolute deadline of the task to simulate any dynamic deadlines.

        Args:
            new_deadline (`EventTime`): The new aboslute deadline (in us) to update
                the task with.

        Raises:
            `ValueError` if the new_deadline < 0.
        """
        if type(new_deadline) != EventTime:
            raise ValueError(
                f"Invalid type received for new_deadline: {type(new_deadline)}"
            )
        if new_deadline < EventTime.zero():
            raise ValueError("Trying to set a negative value for the deadline.")
        self._deadline = new_deadline

    def update_probability(self, new_probability: float):
        """Updates the probability with which the Task is executed.

        Args:
            new_probability (`float`): The new probability to assign to the Task.
        """
        self._probability = new_probability

    def is_complete(self) -> bool:
        """Check if the task has finished its execution.

        To return True, the task must be in either EVICTED / COMPLETED state.

        Returns:
            `True` if the task has finished, `False` otherwise.
        """
        return self.state in (TaskState.EVICTED, TaskState.COMPLETED)

    def is_ready_to_run(self, task_graph: "TaskGraph") -> bool:
        """Check if the task can be placed.

        Args:
            task_graph: The TaskGraph that this Task belongs to.

        Returns:
            `True` if the task is ready to start, `False` otherwise.
        """
        # If the task is terminal, then any one of the parents should
        # be complete, otherwise all should be complete.
        parents_completion_status = [
            parent_task.is_complete() for parent_task in task_graph.get_parents(self)
        ]
        parents_complete = (
            any(parents_completion_status)
            if self.terminal
            else all(parents_completion_status)
        )
        self._logger.debug(
            f"The parent completion status of Task {self} is {parents_complete}."
        )

        # If the parents have finished execution, and the task
        # has been scheduled or preempted, it is ready to run.
        return parents_complete and self.state in (
            TaskState.SCHEDULED,
            TaskState.PREEMPTED,
        )

    def __str__(self):
        if self._logger.isEnabledFor(logging.DEBUG):
            if self.state == TaskState.VIRTUAL:
                if self.release_time == EventTime(-1, EventTime.Unit.US):
                    return (
                        f"Task(name={self.name}, graph={self.task_graph}, "
                        f"id={self.id}, job={self.job}, timestamp={self.timestamp}, "
                        f"state={self.state})"
                    )
                else:
                    if self.deadline == EventTime(-1, EventTime.Unit.US):
                        return (
                            f"Task(name={self.name}, graph={self.task_graph}, "
                            f"id={self.id}, job={self.job}, "
                            f"timestamp={self.timestamp}, state={self.state}, "
                            f"release_time={self.release_time})"
                        )
                    else:
                        return (
                            f"Task(name={self.name}, graph={self.task_graph}, "
                            f"id={self.id}, job={self.job}, "
                            f"timestamp={self.timestamp}, state={self.state}, "
                            f"release_time={self.release_time}, "
                            f"deadline={self.deadline})"
                        )
            elif self.state == TaskState.RELEASED:
                return (
                    f"Task(name={self.name}, graph={self.task_graph}, id={self.id}, "
                    f"job={self.job}, timestamp={self.timestamp}, state={self.state}, "
                    f"release_time={self.release_time}, deadline={self.deadline})"
                )
            elif self.state == TaskState.SCHEDULED or self.state == TaskState.RUNNING:
                return (
                    f"Task(name={self.name}, graph={self.task_graph}, id={self.id}, "
                    f"job={self.job}, timestamp={self.timestamp}, state={self.state}, "
                    f"start_time={self.start_time}, deadline={self.deadline}, "
                    f"remaining_time={self.remaining_time}, "
                    f"worker_pool={self.worker_pool_id})"
                )
            elif self.state == TaskState.CANCELLED:
                return (
                    f"Task(name={self.name}, graph={self.task_graph}, id={self.id}, "
                    f"job={self.job}, timestamp={self.timestamp}, state={self.state}, "
                    f"cancellation_time={self.cancellation_time})"
                )
            elif self.state == TaskState.PREEMPTED:
                return (
                    f"Task(name={self.name}, graph={self.task_graph}, id={self.id}, "
                    f"job={self.job}, timestamp={self.timestamp}, state={self.state}, "
                    f"preemption_time={self.preemption_time}, "
                    f"deadline={self.deadline}, remaining_time={self.remaining_time}, "
                    f"old_worker_pool={self.last_preemption.old_worker_pool})"
                )
            elif self.is_complete():
                return (
                    f"Task(name={self.name}, graph={self.task_graph}, id={self.id}, "
                    f"job={self.job}, timestamp={self.timestamp}, state={self.state}, "
                    f"completion_time={self.completion_time}, deadline={self.deadline})"
                )
        else:
            return self.unique_name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return uuid.UUID(self.id) == uuid.UUID(other.id)

    def __lt__(self, other):
        if self == other:
            return False
        return self.deadline < other.deadline

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_graph(self) -> str:
        return self._task_graph

    @property
    def unique_name(self) -> str:
        return f"{self._name}@{self._task_graph}"

    @cached_property
    def id(self) -> str:
        return str(self._id)

    @property
    def release_time(self) -> EventTime:
        return self._release_time

    @property
    def conditional(self) -> bool:
        return self._creating_job.conditional

    @property
    def probability(self) -> float:
        return self._probability

    @property
    def terminal(self) -> bool:
        return self._creating_job.terminal

    @property
    def intended_release_time(self) -> EventTime:
        return self._intended_release_time

    @property
    def deadline(self) -> EventTime:
        return self._deadline

    @property
    def job(self) -> "Job":  # noqa: F821
        return self._creating_job

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def scheduling_time(self) -> EventTime:
        return self._scheduling_time

    @property
    def cancellation_time(self) -> EventTime:
        return self._cancellation_time

    @property
    def expected_start_time(self) -> EventTime:
        return self._scheduler_placement.placement_time

    @property
    def current_placement(self) -> Placement:
        return self._scheduler_placement

    @property
    def start_time(self) -> EventTime:
        return self._start_time

    @property
    def profile(self) -> WorkProfile:
        return self._profile

    @property
    def available_execution_strategies(self) -> ExecutionStrategies:
        return self._profile.execution_strategies

    @property
    def slowest_execution_strategy(self) -> ExecutionStrategy:
        return self._profile.execution_strategies.get_slowest_strategy()

    @property
    def preemption_time(self) -> EventTime:
        return (
            self.last_preemption.preemption_time
            if self.last_preemption
            else EventTime.invalid()
        )

    @property
    def last_preemption(self) -> Optional["Task.Preemption"]:
        return None if len(self._preemptions) == 0 else self._preemptions[-1]

    @property
    def remaining_time(self) -> EventTime:
        """Retrieves the estimated remaining time according to the state that the Task
        is currently in. The method assumes that the slowest strategy is going to
        execute."""
        if self.state in [TaskState.COMPLETED, TaskState.CANCELLED]:
            return EventTime.zero()
        elif self.state in [
            TaskState.RUNNING,
            TaskState.PREEMPTED,
            TaskState.EVICTED,
            TaskState.SCHEDULED,
        ]:
            return self._remaining_time
        else:
            return self.available_execution_strategies.get_slowest_strategy().runtime

    @property
    def completion_time(self) -> EventTime:
        return self._completion_time

    @property
    def timestamp(self) -> Optional[int]:
        return self._timestamp

    @property
    def worker_pool_id(self) -> Optional[str]:
        return self._worker_pool_id


class TaskGraph(Graph[Task]):
    """A `TaskGraph` represents a directed graph of task dependencies that
    arise due to either the structure of the computation represented in the
    `JobGraph` or the timestamp ordering semantics employed by ERDOS.

    A `TaskGraph` is a runtime entity that constantly evolves as more tasks
    are released by the `JobGraph` and added to the given `TaskGraph`.

    Args:
        name (`str`): The name of this TaskGraph.
        tasks (`Optional[Mapping[Task, Sequence[Task]]]`): An optional
            sequence of mapping between Tasks and their children.
        job_graph (`Optional[JobGraph]`): An optional reference to the JobGraph that
            this TaskGraph is associated with.
    """

    def __init__(
        self,
        name: str,
        tasks: Optional[Mapping[Task, Sequence[Task]]] = {},
        job_graph: Optional["JobGraph"] = None,  # noqa: F821
    ) -> None:
        if type(name) != str:
            raise ValueError("The name must be a string.")
        self._name = name
        self._job_graph = job_graph
        super().__init__(tasks)

    def add_task(self, task: Task, children: Optional[Sequence[Task]] = []):
        """Adds the task to the graph along with the given children.

        Args:
            task (`Task`): The task to be added to the graph.
            children (`Sequence[Task]`): The children of the task, if any.
        """
        self.add_node(task, *children)

    def cancel(self, task: Task, time: EventTime) -> Sequence[Task]:
        """Cancels a task along with any tasks that cannot execute as a result
        of this task's cancellation. Note that this method also cancels the
        execution of the task thus removing the need to call cancel on the task
        independently.

        Args:
            task (`Task`): The task that needs to be cancelled along with its
                descendants.
            time (`EventTime`): The time at which the tasks need to be cancelled.

        Returns:
            The set of tasks that were cancelled as a result of the cancellation
            of this task (includes the calling task).
        """
        # Do a breadth-first traversal from this task to all the children of this
        # task until the first terminal task, or until no task is left.
        cancelled_tasks = []
        for child in self.depth_first(task):
            task._logger.debug(
                "[%s] The TaskGraph %s is considering whether to cancel Task %s.",
                time.to(EventTime.Unit.US).time,
                self.name,
                child.unique_name,
            )
            if (
                child.terminal
                and child != task
                and not all(
                    parent.state == TaskState.CANCELLED
                    for parent in self.get_parents(child)
                )
            ):
                # If the terminal child cannot be cancelled because one or more
                # of its parents can still finish execution, then stop, otherwise
                # continue the cancellation of children.
                # Note: We only use this check to ensure that we do not inadvertently
                # cancel a terminal task as a result of a cascading cancellation.
                # However, if the cancellation of the terminal task is requested, then
                # we comply with the request.
                task._logger.debug(
                    "[%s] The cancellation of the Task %s and any future tasks was "
                    "stopped because %s was a terminal task and the state of its "
                    "parents was: %s.",
                    time.to(EventTime.Unit.US).time,
                    child.unique_name,
                    child.unique_name,
                    str(
                        [
                            (parent.unique_name, parent.state)
                            for parent in self.get_parents(child)
                        ]
                    ),
                )
                break

            if child.state == TaskState.CANCELLED:
                # This child (and all its children) were cancelled, stop execution.
                task._logger.debug(
                    "[%s] The Task %s was in the state %s, and no future children "
                    "are being cancelled.",
                    time.to(EventTime.Unit.US).time,
                    child.unique_name,
                    child.state,
                )
                break

            cancelled_tasks.append(child)
            child.cancel(time)

        return cancelled_tasks

    def notify_task_completion(
        self, task: Task, finish_time: EventTime
    ) -> Tuple[Sequence[Task], Sequence[Task]]:
        """Notify the completion of the task.

        The caller must set the type of the task completion before invoking
        this method to ensure that the proper dependencies are unlocked.

        Args:
            task (`Task`): The task that has finished execution.
            finish_time (`EventTime`): The time (in us) at which the task finished.

        Returns:
            A tuple with the set of tasks released by the completion of this task,
            along with the set of tasks cancelled by the completion of this task.

        Raises:
            `ValueError` if an incomplete task is passed.
        """
        if type(finish_time) != EventTime:
            raise ValueError(
                f"Invalid type received for finish_time: {type(finish_time)}"
            )
        # Ensure that the task is actually complete.
        if not task.is_complete():
            raise ValueError(f"Cannot notify TaskGraph of an incomplete task: {task}")

        # Release any tasks that can be unlocked by the completion.
        # If the task was conditional, only release one of the children randomly.
        released_tasks = []

        # Inform the caller of any tasks that were cancelled byt the completion of this
        # task. If the task was conditional, all the children on the path that was not
        # released will be cancelled by this task.
        cancelled_tasks = []

        if task.conditional:
            # Choose a child randomly from the set of children.
            # Ensure that the probability of the children is adding up to 1.
            task_children = self.get_children(task)
            task_children_probabilities = [child.probability for child in task_children]
            if all(
                [prob <= sys.float_info.epsilon for prob in task_children_probabilities]
            ):
                # The task was conditional, and no child was available for execution.
                # Cancel all children tasks since we won't be running them. This issue
                # only occurs when we use the resolve_conditionals_at_submission flag,
                # since it sets a zero probability of occurrence of some tasks.
                for child in task_children:
                    cancelled_tasks.extend(self.cancel(child, finish_time))
                return (released_tasks, cancelled_tasks)
            if abs(sum(task_children_probabilities) - 1.0) > sys.float_info.epsilon:
                raise ValueError(
                    f"The sum of the probability of children of {task.unique_name}"
                    f" exceeds 1.0: {task_children_probabilities}."
                )

            # Choose a child to release using the probability distribution.
            child_to_release = random.choices(
                population=task_children, weights=task_children_probabilities, k=1
            )[0]
            if (
                child_to_release.state > TaskState.SCHEDULED
                and child_to_release.state < TaskState.CANCELLED
            ):
                # Task should not move past the scheduled state until its
                # last parent finishes.
                raise RuntimeError(
                    f"Child task {child_to_release} moved beyond SCHEDULED state "
                    f"without completion of the parent {task}."
                )

            # Set the child's probability to 1.0 now that a decision has been made.
            # Do a breadth first search until the first terminal node from the children
            # whose branch is not taken, and cancel the tasks.
            for child in task_children:
                if child == child_to_release:
                    child.update_probability(1.0)
                else:
                    cancelled_tasks.extend(self.cancel(child, finish_time))

            # Request the Simulator to release the tasks.
            released_tasks.append(child_to_release)
        else:
            for child in self.get_children(task):
                if (
                    child.state > TaskState.SCHEDULED
                    and child.state < TaskState.CANCELLED
                ):
                    # Task should not move past the scheduled state until its
                    # last parent finishes.
                    raise RuntimeError(
                        f"Child task {child} moved beyond SCHEDULED state "
                        f"without completion of the parent {task}."
                    )
                elif child.state == TaskState.CANCELLED:
                    # If the future task has been cancelled, then we skip
                    # releasing the task to prevent blocking the simulator.
                    continue
                if child.terminal or all(
                    map(lambda task: task.is_complete(), self.get_parents(child))
                ):
                    # If the child was a terminal of a conditional, then only one
                    # of the parents being released should release the task.
                    # We release the task now since one parent has completed.
                    released_tasks.append(child)
        return released_tasks, cancelled_tasks

    def find(self, task_name: str) -> Sequence[Task]:
        """Find all the instances of a task with the given name.

        Use this method to retrieve the instances of a task from the graph,
        and query / change their parameters.

        Args:
            task_name (`str`): Find a task with the given name.

        Returns:
            A possibly empty `Sequence[Task]` with the given name.
        """
        return self.filter(lambda task: task.name == task_name)

    def get_schedulable_tasks(
        self,
        time: EventTime,
        lookahead: EventTime = EventTime.zero(),
        preemption: bool = False,
        retract_schedules: bool = False,
        worker_pools: "WorkerPools" = None,  # noqa: F821
        policy: BranchPredictionPolicy = BranchPredictionPolicy.ALL,
        branch_prediction_accuracy: float = 0.50,
        release_taskgraphs: bool = False,
        debug: bool = False,
    ) -> Sequence[Task]:
        """Retrieves all tasks from the given TaskGraph that are expected to be
        available for scheduling within the horizon defined by `time + lookahead`.

        The method propagates the expected start times of each of the tasks in the
        TaskGraph according to the given release times or the completion times of the
        parent tasks. The tasks that are expected to be available for scheduling within
        the horizon are then chosen, and previously scheduled or running tasks are
        considered for schedulability if `retract_schedules` or `preemption` is
        enabled.

        When constructing the estimated completion times of the tasks, the value of
        the conditionals is resolved according to the specified `policy`. By default,
        all possible conditional paths are considered for execution, and the maximum
        estimated release time is then chosen for the child tasks.

        Args:
            time (`EventTime`): The time at which the schedulable tasks are being
                retrieved (the current simulator time).
            lookahead (`EventTime`): The added time upto which the schedulable tasks
                need to be retrieved (i.e. time + lookahead).
            preemption (`bool`): If `True`, ddd currently running tasks to the set of
                schedulable tasks.
            retract_schedules (`bool`): If `True`, allow already scheduled tasks to be
                reconsidered for schedulability.
            worker_pools (`WorkerPools`): A representation of the `WorkerPools` that
                are currently being used for scheduling. This object is only used if
                preemption is enabled.
            policy (`BranchPredictionPolicy`): The branch prediction policy to use when
                deciding what tasks can be considered in the scheduling horizon.
            branch_prediction_accuracy (`float`): The accuracy with which the branches
                should be correctly predicted by the TaskGraph.
            release_taskgraphs (`bool`): If `True`, all tasks of the `TaskGraph` are
                made available for scheduling if any task in the `TaskGraph` fall within
                the scheduler lookahead. This option uses the `policy` to decide what
                tasks to release if they are made available for scheduling.
            debug (`bool`): If `True`, print debug information about the decision
                making.

        Returns:
            A list of tasks that are schedulable in the `time + lookahead` horizon.
        """
        # Estimate the completion time of materialized tasks.
        task_queue = deque([])
        estimated_completion_time = {}
        for task in self.get_nodes():
            if task.state == TaskState.COMPLETED:
                estimated_completion_time[task] = task.completion_time
            elif task.state in [
                TaskState.RUNNING,
                TaskState.PREEMPTED,
                TaskState.EVICTED,
            ]:
                # Optimistically assume that the task will be resumed right now.
                estimated_completion_time[task] = time + task.remaining_time
            elif task.state == TaskState.RELEASED:
                estimated_completion_time[task] = (
                    max(task.release_time, time) + task.remaining_time
                )
            elif task.state == TaskState.SCHEDULED:
                if retract_schedules:
                    # Assume that the task will be placed again now.
                    slowest_strategy = (
                        task.available_execution_strategies.get_slowest_strategy()
                    )
                    estimated_completion_time[task] = time + slowest_strategy.runtime
                else:
                    estimated_completion_time[task] = (
                        task.expected_start_time + task.remaining_time
                    )
            elif task.state == TaskState.CANCELLED:
                # Completion times of cancelled tasks is not estimated.
                continue
            elif task.state == TaskState.VIRTUAL:
                # There's no way to estimate the completion time of a task
                # in this state from known values, propagate completion times
                # from the parents here.
                continue
            else:
                raise ValueError(
                    f"Task {task.unique_name} in unknown state: {task.state}."
                )
            task_queue.append(task)
            if debug:
                task._logger.debug(
                    f"[{time.to(EventTime.Unit.US).time}] After considering the "
                    f"task {task.unique_name}, the task queue is "
                    f"{[_task.unique_name for _task in task_queue]}."
                )

        # Estimate the completion time of VIRTUAL tasks.
        while len(task_queue) > 0:
            task = task_queue.popleft()
            completion_time = estimated_completion_time[task]

            # Find the children tasks to propogate the completion time
            # to according to the policy.
            children_tasks = []
            if task.conditional:
                children_tasks = self.resolve_conditional(
                    task, policy, branch_prediction_accuracy
                )
            else:
                children_tasks = self.get_children(task)

            for child_task in children_tasks:
                if (
                    not retract_schedules and child_task.state != TaskState.VIRTUAL
                ) or (
                    retract_schedules
                    and child_task.state not in (TaskState.VIRTUAL, TaskState.SCHEDULED)
                ):
                    # Skip the task because we've already set its completion time.
                    continue

                # Compute the estimated completion time of the child task.
                # If the child task was provided with a specific release time, and
                # that time leads to a later completion time, use that completion time.
                slowest_strategy = (
                    child_task.available_execution_strategies.get_slowest_strategy()
                )
                child_completion_time = completion_time + slowest_strategy.runtime
                if child_task.release_time:
                    child_completion_time = max(
                        child_completion_time,
                        child_task.release_time + slowest_strategy.runtime,
                    )

                # Update the completion time of the child.
                if (
                    child_task not in estimated_completion_time
                    or child_completion_time > estimated_completion_time[child_task]
                ):
                    estimated_completion_time[child_task] = child_completion_time
                    task_queue.append(child_task)

        if debug:
            estimated_completion_time_output = [
                f"{_task.unique_name} ({_completion_time})"
                for _task, _completion_time in estimated_completion_time.items()
            ]
            task._logger.debug(
                f"[{time.to(EventTime.Unit.US).time}] The set of tasks with their "
                f"available estimated completion times are: "
                f"{estimated_completion_time_output}."
            )

        # Add the tasks conforming to the policy within the lookahead.
        tasks = []

        # Track if any released tasks have been added to the set of available
        # tasks. If yes, and the `release_taskgraphs` option was set, then
        # forego the checking of the estimated completion time for the virtual
        # tasks. This works in a loop since we iterate on the tasks in the
        # topologically-sorted order.
        any_released = False
        for task in self.topological_sort():
            if task.state in (TaskState.COMPLETED, TaskState.RUNNING):
                # The task has already been completed or is running,
                # we should notify that all tasks of this TaskGraph
                # must be released now if required.
                any_released = True
            elif (
                task.state == TaskState.RELEASED
                and task.release_time <= time + lookahead
            ):
                # The Task has already been released and its release time is within
                # the lookahead.
                tasks.append(task)
                any_released = True
            elif task.state in (TaskState.PREEMPTED, TaskState.EVICTED):
                # PREEMPTED and EVICTED tasks are always available for scheduling.
                tasks.append(task)
                any_released = True
            elif (
                task.state == TaskState.VIRTUAL
                and task in estimated_completion_time
                and (
                    (any_released and release_taskgraphs)
                    or estimated_completion_time[task]
                    <= time + lookahead + task.remaining_time  # Check the release time
                )
            ):
                # A VIRTUAL task that may be available for scheduling.
                tasks.append(task)
                any_released = True
            elif (
                retract_schedules
                and task.state == TaskState.SCHEDULED
                and task in estimated_completion_time
                and (
                    (any_released and release_taskgraphs)
                    or estimated_completion_time[task]
                    <= time
                    + lookahead
                    + task.available_execution_strategies.get_slowest_strategy().runtime
                )
            ):
                # A SCHEDULED task that is being reconsidered for scheduling.
                tasks.append(task)
                any_released = True
            if debug:
                task._logger.debug(
                    f"[{time.to(EventTime.Unit.US).time}] After considering "
                    f"{task.unique_name} ({task.state}), the set of potentially "
                    f"schedulable tasks are {[_task.unique_name for _task in tasks]},"
                    f" and the status of any_released is {any_released}."
                )

        # No need to add already running tasks if preemption is not enabled.
        if preemption:
            if worker_pools:
                # Adding the tasks placed on the given worker pool.
                tasks.extend(worker_pools.get_placed_tasks())
            else:
                # No worker pool provided. Getting all RUNNING tasks.
                tasks.extend(
                    self.filter(
                        lambda task: task.state
                        in (TaskState.SCHEDULED, TaskState.RUNNING)
                    )
                )

        return tasks

    def get_releasable_tasks(self) -> Sequence[Task]:
        """Retrieves the set of Tasks that have no dependencies and are thus available
        to run.

        Returns:
            A list of tasks that can be released.
        """
        tasks_to_be_released = []
        for task in self.get_nodes():
            if task.state not in RELEASABLE_TASK_STATES:
                continue
            parents = self.get_parents(task)
            if all(map(lambda task: task.is_complete(), parents)):
                tasks_to_be_released.append(task)
        return tasks_to_be_released

    def clean(self):
        """Cleans the `TaskGraph` of tasks that have finished completion.

        This is supposed to be called at regular intervals by the simulator so
        we don't end up a huge graph of finished tasks.

        We only clean up the tasks whose children have finished execution.
        """
        tasks_to_clean = []
        for task in self.get_nodes():
            # Has this task finished execution?
            if task.is_complete():
                # Check if all children have finished execution too.
                can_be_cleaned = True
                for child in self.get_children(task):
                    if not child.is_completed():
                        can_be_cleaned = False
                        break

                # If the task can be cleaned, add the entry to be removed.
                if can_be_cleaned:
                    tasks_to_clean.append(task)

        # Remove the task from the parent graph of all its children, and then
        # remove the task from the graph itself.
        for task in tasks_to_clean:
            self.remove(task)

    def __getitem__(self, slice_obj) -> Union["TaskGraph", Sequence["TaskGraph"]]:
        """Retrieve a slice of the TaskGraph as specified by the `slice_obj`.

        The `slice_obj` should specify the timestamps for which the slice is
        required. In case a slice object is passed, this method returns a
        `Sequence[TaskGraph]` of sliced TaskGraphs with the corresponding
        timestamps.
        """
        if isinstance(slice_obj, int):
            # Get the slice for a single timestamp.
            tasks = {}
            for task in self.filter(lambda task: task.timestamp == slice_obj):
                # Maintain the task dependencies with the same timestamps.
                same_timestamp_children = []
                for child in self.get_children(task):
                    if child.timestamp == slice_obj:
                        same_timestamp_children.append(child)

                # Add the task to the representation.
                tasks[task] = same_timestamp_children
            return TaskGraph(name=f"{self.name}_slice", tasks=tasks)
        elif isinstance(slice_obj, slice):
            max_timestamp = -sys.maxsize
            for task in self.get_nodes():
                if task.timestamp > max_timestamp:
                    max_timestamp = task.timestamp
            return [
                self[index] for index in range(*slice_obj.indices(max_timestamp + 1))
            ]
        else:
            raise ValueError(f"Unexpected value while slicing: {slice_obj}")

    def is_source_task(self, task: Task) -> bool:
        """Check if the given `task` is a source Task or not.

        Args:
            task (`Task`): The task to check.

        Returns:
            `True` if the task is a source task i.e. only has a dependency on
            the same task of the previous timestamp, and `False` otherwise.
        """
        parents = self.get_parents(task)
        return len(parents) == 0 or (
            len(parents) == 1
            and parents[0].name == task.name
            and parents[0].timestamp == task.timestamp - 1
        )

    def is_sink_task(self, task: Task) -> bool:
        """Check if the given `task` is a sink Task or not.

        Args:
            task (`Task`): The task to check.

        Returns:
            `True` if the task is a sink task i.e. only has a dependency on the same
            task of the next timestamp, and `False` otherwise.
        """
        children = self.get_children(task)
        return len(children) == 0 or (
            len(children) == 1
            and children[0].name == task.name
            and children[0].timestamp == task.timestamp + 1
        )

    def get_source_tasks(self) -> Sequence[Task]:
        """Retrieve the source tasks from this instance of a TaskGraph.

        This method returns multiple instances of Tasks with different
        timestamps.

        Returns:
            A `Sequence[Task]` of tasks that have no dependencies on any
            tasks with the same timestamps.
        """
        return self.filter(self.is_source_task)

    def get_sink_tasks(self) -> Sequence[Task]:
        """Retrieve the sink tasks from the instance of the TaskGraph.

        This method returns multiple instances of Tasks with different
        timestamps.

        Returns:
            A `Sequence[Task]` of tasks that have no dependencies on any
            tasks with the same timestamps.
        """
        return self.filter(self.is_sink_task)

    def dilate(self, difference: EventTime):
        """Dilate the time between occurrence of events of successive
        logical timestamps according to the given difference.

        If the provided difference is greater than the time between the
        occurrence of two events, the effect is a slowdown of the events.

        If the provided difference is smaller than the time between the
        occurrence of two events, the effect is a speedup of the events.

        This method changes the `TaskGraph` in-place by modifying the release
        time of the actual tasks.

        Args:
            difference (`EventTime`): The time difference (in us) to keep between the
                occurrence of two source Tasks of different timestamps.
        """
        if type(difference) != EventTime:
            raise ValueError(
                f"Invalid type received for difference: {type(difference)}"
            )
        for parent_graph, child_graph in zip(self[0:], self[1:]):
            # Fix the offsets of the source tasks according to the release
            # time of the parents.
            child_source_tasks = child_graph.get_source_tasks()
            offsets = []
            for child_source_task in child_source_tasks:
                parent_source_task = parent_graph.find(child_source_task.name)
                assert (
                    len(parent_source_task) == 1
                ), f"Expected a single parent for: {child_source_task}"
                parent_source_task = parent_source_task[0]

                # Calculate the offset and set the release time according to
                # the parent task's release time and the provided difference.
                offset = child_source_task.release_time - (
                    parent_source_task.release_time + difference
                )
                offsets.append(offset)
                child_source_task._release_time -= offset
                child_source_task._deadline -= offset

            # Calculate the average of the offsets of the source tasks and
            # offset the remainder of the tasks by the average.
            summed_offset = sum(offsets, start=EventTime.zero())
            average_offset = int(summed_offset.time / len(offsets))
            for task in child_graph.get_nodes():
                if not child_graph.is_source_task(task):
                    task._release_time -= EventTime(average_offset, summed_offset.unit)
                    task._deadline -= EventTime(average_offset, summed_offset.unit)

    def merge(self, task_graphs: Sequence["TaskGraph"]) -> "TaskGraph":
        """Merge the given task_graphs after ordering them by timestamps.

        Args:
            task_graphs (`Sequence[TaskGraph]`): A sequence of task graphs to
                be merged.

        Returns:
            The merged TaskGraph from the sequence ordered by timestamp.
        """
        raise NotImplementedError("Merging of Taskgraphs has not been implemented yet.")

    def is_scheduled(self) -> bool:
        """Check if any of the tasks in the TaskGraph have been scheduled.

        Returns:
            `True` if any of the tasks in the TaskGraph have been scheduled,
            and `False` otherwise.
        """
        return any(task.state == TaskState.SCHEDULED for task in self.get_nodes())

    def is_complete(self) -> bool:
        """Check if the task graph has finished execution.

        Returns:
            `True` if all the sink tasks in the TaskGraphs have finished execution,
            and `False` otherwise.
        """
        return all(task.is_complete() for task in self.get_sink_tasks())

    def is_cancelled(self) -> bool:
        """Check if the task graph has been cancelled, and will not finish execution.

        Returns
            `True` if any of the sink tasks in the TaskGraph have been cancelled, and
            `False` otherwise.
        """
        return any(task.state == TaskState.CANCELLED for task in self.get_sink_tasks())

    def resolve_conditional(
        self,
        task: Task,
        policy: BranchPredictionPolicy,
        branch_prediction_accuracy: float = 0.50,
    ) -> Sequence[Task]:
        """Resolves the conditional of the given `task` according to the
        specified `policy`.

        Args:
            task (`Task`): The Task to resolve the conditional for.
            policy (`BranchPredictionPolicy`): The policy to use for resolving the
                conditionals.
            branch_prediction_accuracy (`float`): The accuracy with which to predict
                the branch to be taken if the policy is RANDOM.

        Returns:
            A `Sequence[Task]` that specifies the children to be executed
            according to the specified `policy`.
        """
        if not task.conditional:
            raise ValueError(f"The task {task} was not conditional.")

        resolved_tasks = []
        children_tasks = self.get_children(task)
        if task.is_complete():
            # Choose the branch that was actually executed.
            released_child = children_tasks[0]
            for child in children_tasks[1:]:
                if child.probability > released_child.probability:
                    released_child = child
            resolved_tasks.append(released_child)
        elif policy == BranchPredictionPolicy.WORST_CASE:
            # Choose the branch that has the lowest probability.
            child_to_release = children_tasks[0]
            for child in children_tasks[1:]:
                if child.probability < child_to_release.probability:
                    child_to_release = child
            resolved_tasks.append(child_to_release)
        elif policy == BranchPredictionPolicy.BEST_CASE:
            # Choose the branch that has the highest probability.
            child_to_release = children_tasks[0]
            for child in children_tasks[1:]:
                if child.probability > child_to_release.probability:
                    child_to_release = child
            resolved_tasks.append(child_to_release)
        elif policy == BranchPredictionPolicy.MAXIMUM:
            # Choose the branch that has the highest runtime.
            # TODO (Sukrit): This branch may not have the highest resource requirement.
            # TODO (Sukrit): The runtime of this branch is calculated as a depth-first
            # traversal, and not a sum of all branches.
            maximum_branch_time = EventTime.zero()
            child_to_release = children_tasks[0]
            for child in children_tasks:
                branch_time = EventTime.zero()
                for node in self.depth_first(child):
                    if node.terminal:
                        break
                    branch_time += node.remaining_time

                if branch_time > maximum_branch_time:
                    maximum_branch_time = branch_time
                    child_to_release = child
            resolved_tasks.append(child_to_release)
        elif policy == BranchPredictionPolicy.RANDOM:
            # Check if any of the children have already moved to SCHEDULED state, and
            # just return that for consistency.
            for child in children_tasks:
                if child.state >= TaskState.SCHEDULED:
                    resolved_tasks.append(child)
                    return resolved_tasks

            # None of the tasks have been SCHEDULED yet, check if the conditionals
            # were resolved at submission. If yes, use the branch_prediction_accuracy
            # to decide if we should release that task or something else.
            if all(
                abs(child.probability - 1.0) < sys.float_info.epsilon
                or child.probability < sys.float_info.epsilon
                for child in children_tasks
            ):
                # The tasks were resolved at submission. Find the task that is going
                # to be released.
                child_to_release = children_tasks[0]
                for child in children_tasks[1:]:
                    if child.probability > child_to_release.probability:
                        child_to_release = child

                # If the outcome of a biased coin is less than the required probability,
                # then we release the correct task, and choose a random task otherwise.
                if random.random() < branch_prediction_accuracy:
                    resolved_tasks.append(child_to_release)
                else:
                    # Collect all the other tasks that will not be released.
                    other_tasks = []
                    for child in children_tasks:
                        if child != child_to_release:
                            other_tasks.append(child)
                    resolved_tasks.append(random.choice(other_tasks))
            else:
                resolved_tasks.append(random.choice(children_tasks))
        elif policy == BranchPredictionPolicy.ALL:
            # Release all children tasks.
            resolved_tasks.extend(children_tasks)
        else:
            raise ValueError(f"The policy {policy} is not supported.")
        return resolved_tasks

    def get_remaining_time(
        self, policy: BranchPredictionPolicy = BranchPredictionPolicy.ALL
    ):
        """Retrieves the time remaining in the completion of the TaskGraph.

        By default, the method returns the worst-case remaining time (i.e., the
        maximum of remaining time across all conditional branches).

        Returns:
            An `EventTime` denoting the remaining time computed according to the
            provided `policy`.
        """
        remaining_time = defaultdict(lambda: EventTime.zero())

        # Add the remaining time for the sources.
        for source_task in self.get_source_tasks():
            remaining_time[source_task] = source_task.remaining_time

        # Iterate over the other nodes and choose the remaining
        # time according to the policy.
        for task in self.topological_sort():
            # The branch associated with this task was not executed, skipping.
            if task not in remaining_time:
                continue

            # Propogate the remaining time to the children chosen by either
            # the policy (in case of conditionals), or all the children otherwise.
            children_tasks = None
            if task.conditional:
                children_tasks = self.resolve_conditional(task, policy)
            else:
                children_tasks = self.get_children(task)
            for child_task in children_tasks:
                if (
                    remaining_time[child_task]
                    <= remaining_time[task] + child_task.remaining_time
                ):
                    remaining_time[child_task] = (
                        remaining_time[task] + child_task.remaining_time
                    )

        # Find the maximum remaining time across all the sink nodes.
        return max([remaining_time[sink] for sink in self.get_sink_tasks()])

    def get_task(self, name: str, unique: bool = False) -> Optional[Task]:
        """Retrieve the Task with the given name from the TaskGraph.

        Returns `None` if no such Task exists in the TaskGraph.

        Args:
            name (`str`): The name of the Task to retrieve.
            unique (`bool`): If `True`, then the Task will be matched by its unique
                name instead of the name.

        Returns:
            The `Task` with the given name, if it exists in the TaskGraph.

        Raises:
            `ValueError` if multiple tasks with the same name are found.
        """
        matched_task = None
        for task in self.get_nodes():
            if task.name == name or (unique and task.unique_name == name):
                if matched_task:
                    raise ValueError(f"Multiple tasks with the name {name} found.")
                matched_task = task
        return matched_task

    def update_edges(self, tasks: Mapping[Task, Sequence[Task]]) -> None:
        """Updates the edges inside the TaskGraph according to the new set of tasks.

        This is used to transform the TaskGraph without changing the references that
        may be held by other objects to this TaskGraph.

        Args:
            tasks (`Mapping[Task, Sequence[Task]]`): A mapping of tasks to their
                children tasks.
        """
        super(TaskGraph, self).__init__(tasks)

    @property
    def deadline(self) -> EventTime:
        """Retrieve the deadline to which the TaskGraph is being subjected to.

        This is simply the maximum of all the deadlines of the Tasks.

        Returns:
            An `EventTime` denoting the maximum deadline of all the Tasks.
        """
        return max(task.deadline for task in self.get_nodes())

    @cached_property
    def critical_path_runtime(self) -> EventTime:
        """Retrieve the runtime of the critical path of the TaskGraph.

        This runtime is calculated by assuming the slowest ExecutionStrategy for all
        the nodes in the TaskGraph.

        Returns:
            An `EventTime` denoting the runtime of the critical path of the TaskGraph.
        """
        return EventTime(
            sum(
                task.slowest_execution_strategy.runtime.to(EventTime.Unit.US).time
                for task in self.get_longest_path(
                    lambda t: t.slowest_execution_strategy.runtime.to(
                        EventTime.Unit.US
                    ).time
                )
            ),
            EventTime.Unit.US,
        )

    @property
    def completion_time(self) -> EventTime:
        """Retrieve the completion time of the TaskGraph.

        The completion time of the TaskGraph is the maximum of the completion time
        of all of its sink nodes. If the sink nodes haven't been completed yet, then
        this method will raise a RuntimeError.

        Returns:
            An `EventTime` representation of the completion time of the TaskGraph.
        """
        if not self.is_complete():
            raise RuntimeError(
                f"The tasks of the Graph {self.name} are not complete yet."
            )
        return max(sink_task.completion_time for sink_task in self.get_sink_tasks())

    @property
    def release_time(self) -> EventTime:
        """Retrieves the time at which the earliest Task in this
        TaskGraph will be released.

        Returns:
            An `EventTime` representation of the release time of the TaskGraph.
        """
        return min(task.release_time for task in self.get_source_tasks())

    @property
    def name(self) -> str:
        """Retrieves the name of the TaskGraph.

        Returns:
            A `str` representation of the name of the TaskGraph.
        """
        return self._name

    @property
    def job_graph(self) -> Optional["JobGraph"]:  # noqa: F821
        """Retrieves the JobGraph associated with the TaskGraph.

        Returns:
            A `JobGraph` object representing the JobGraph associated with the TaskGraph.
        """
        return self._job_graph

    def __str__(self):
        constructed_string = ""
        for task in iter(self):
            constructed_string += "{}_{}: {}\n".format(
                task.name,
                task.timestamp,
                list(
                    map(
                        lambda t: "{}_{}".format(t.name, t.timestamp),
                        self.get_children(task),
                    )
                ),
            )
        return constructed_string
