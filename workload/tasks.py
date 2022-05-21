import logging
import random
import sys
import uuid
from collections import defaultdict, deque, namedtuple
from enum import Enum
from functools import total_ordering
from typing import Mapping, Optional, Sequence, Union

import utils
from workload import Job, Resources
from workload.graph import Graph

Preemption = namedtuple(
    "Preemption", "preemption_time, old_worker_pool, restart_time, new_worker_pool"
)


class TaskState(Enum):
    """Represents the different states that a Task can potentially be in."""

    VIRTUAL = 1  # The Task is expected to be generated in the future.
    RELEASED = 2  # The Task has been released, and is waiting to be run.
    RUNNING = 3  # The Task has begun execution, and is currently running.
    PREEMPTED = 4  # The Task had begun execution but is currently paused.
    EVICTED = 5  # The Task has been evicted before completing.
    COMPLETED = 6  # The Task has successfully completed.


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
        job (`Job`): The job that created this particular task.
        resource_requirements (`Resources`): The set of resources required by
            this task.
        release_time (`int`): The time at which the task was released by the
            job (in us).
        runtime (`int`): The expected runtime (in us) of this task.
        deadline (`int`): The absolute deadline by which the task should complete.
        state (`TaskState`): Defines the state of the task.
        timestamp (`int`): The timestamp for the Task (single dimension).
        start_time (`int`): The time (in us) at which the task was started
            (only available if state != TaskState.RELEASED, -1 otherwise)
        remaining_time (`int`): The time (in us) remaining to finish the
            completion of the task.
        completion_time (`int`): The time (in us) at which the task completed
            / was preempted (only available if state is either
            EVICTED / COMPLETED, -1 otherwise)
        _logger(`Optional[logging.Logger]`): The logger to use to log the
            results of the execution.
    """

    def __init__(
        self,
        name: str,
        job: Job,
        resource_requirements: Resources,
        runtime: int,
        deadline: int,
        timestamp: int = None,
        release_time: Optional[int] = -1,
        start_time: Optional[int] = -1,
        completion_time: Optional[int] = -1,
        _logger: Optional[logging.Logger] = None,
    ):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=f"{name}_{timestamp}")

        self._name = name
        self._creating_job = job
        self._resource_reqs = resource_requirements
        self._expected_runtime = runtime
        self._deadline = deadline
        self._timestamp = timestamp
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)

        # The timestamps maintained for each state of the task.
        # (VIRTUAL -> RELEASED)
        self._intended_release_time = release_time
        self._release_time = release_time
        # (RELEASED -> RUNNING)
        self._start_time = start_time
        # (RUNNING -> EVICTED / COMPLETED)
        self._completion_time = completion_time
        # (RUNNING -> PREEMPTED)
        self._preemptions = []

        # The data required for managing the execution of a particular task.
        self._remaining_time = runtime
        self._last_step_time = -1  # Time when this task was stepped through.
        self._state = TaskState.VIRTUAL
        # ID of the worker pool on which the task is running.
        self._worker_pool_id = None

    def release(self, time: Optional[int] = None):
        """Release the task and transition away from the virtual state.

        Args:
            time (`Optional[int]`): The simulation time (in us) at which to
                release the task. If None, should be specified at task
                construction.
        """
        self._logger.debug(f"Transitioning {self} to {TaskState.RELEASED}")
        if time is None and self._release_time == -1:
            raise ValueError(
                "Release time should be specified either while "
                "creating the Task or when releasing it."
            )
        if self.state != TaskState.VIRTUAL:
            raise ValueError(f"Cannot release {self.id} which is in state {self.state}")
        if time is not None:
            self._release_time = time

        self._state = TaskState.RELEASED

    def start(
        self,
        time: Optional[int] = None,
        variance: Optional[int] = 0,
    ):
        """Begins the execution of the task at the given simulator time.

        Args:
            time (`Optional[int]`): The simulation time (in us) at which to
                begin the task. If None, should be specified at task
                construction.
            variance (`Optional[int]`): The percentage variation to add to
                the runtime of the task.

        Raises:
            `ValueError` if Task is not in `RELEASED`/`PREEMPTED` state yet.
        """
        if self.state not in (TaskState.RELEASED, TaskState.PREEMPTED):
            raise ValueError(
                f"Only RELEASED or PREEMPTED tasks can be started. "
                f"Task is in state {self.state}"
            )
        if time is None and self._start_time == -1:
            raise ValueError(
                "Start time should be specified either while "
                "creating the Task or when starting it."
            )

        remaining_time = utils.fuzz_time(self._remaining_time, (0, variance))
        self._logger.debug(
            f"Transitioning {self} to {TaskState.RUNNING} at time {time} "
            f"with the remaining time {remaining_time}"
        )
        self._start_time = time if time is not None else self._start_time
        assert (
            self._start_time >= self._release_time
        ), f"Task {self.id} start time must be greater than release time"
        self._last_step_time = time
        self._state = TaskState.RUNNING
        self.update_remaining_time(remaining_time)

    def step(self, current_time: int, step_size: int = 1) -> bool:
        """Steps the task for the given `step_size` (default 1 time step).

        Args:
            current_time (`in`): The current time of the simulator (in us).
            step_size (`int`): The amount of time (in us) for which to step
                the task.

        Returns:
            `True` if the task has finished execution, `False` otherwise.
        """
        if (
            self.state != TaskState.RUNNING
            or self.start_time > current_time + step_size
        ):
            # We cannot step a Task that's not supposed to be running.
            self._logger.warning(
                f"Cannot step {self} with start time {self.start_time} at "
                f"time {current_time} since it's either not RUNNING or isn't "
                f"supposed to start yet."
            )
            return False

        # Task can be run, step through the task's execution.
        execution_time = current_time + step_size - self._last_step_time
        if self._remaining_time - execution_time <= 0:
            self._last_step_time = current_time + self._remaining_time
            self._remaining_time = 0
            self.finish(self._last_step_time)
            return True
        else:
            self._last_step_time = current_time + step_size
            self._remaining_time -= execution_time
            self._logger.debug(
                f"Stepped {self} for {step_size} steps. "
                f"Remaining execution time: {self._remaining_time}"
            )
            return False

    def preempt(self, time: int):
        """Preempts the task at the given simulation time.

        Args:
            time (`int`): The simulation time (in us) at which to preempt the task.

        Raises:
            `ValueError` if task is not RUNNING.
        """
        if self.state != TaskState.RUNNING:
            raise ValueError(f"Task {self.id} is not RUNNING right now.")
        self._logger.debug(
            f"Transitioning {self} to {TaskState.PREEMPTED} at time {time}"
        )
        self._preemptions.append(
            Preemption(
                preemption_time=time,
                old_worker_pool=self._worker_pool_id,
                restart_time=None,
                new_worker_pool=None,
            )
        )
        self._state = TaskState.PREEMPTED
        self._worker_pool_id = None

    def resume(self, time: int, worker_pool_id: Optional[str] = None):
        """Continues the execution of the task at the given simulation time.

        If the `worker_pool_id` passed is `None`, it is assumed that the task
        will be restarted at the old worker pool.

        Args:
            time (`int`): The simulation time (in us) at which to restart the
                task.
            worker_pool_id (`Optional[str]`): The ID of the WorkerPool that
                the task will be resumed on.

        Raises:
            `ValueError` if task is not PREEMPTED.
        """
        if self.state != TaskState.PREEMPTED:
            raise ValueError(f"Task {self.id} is not PREEMPTED right now.")
        new_worker_pool = (
            worker_pool_id if worker_pool_id else self.last_preemption.old_worker_pool
        )
        self._logger.debug(
            f"Transitioning {self} which was PREEMPTED at "
            f"{self.preemption_time} to {TaskState.RUNNING} at "
            f"time {time} on WorkerPool ({new_worker_pool})"
        )
        self.last_preemption._replace(restart_time=time)
        self.last_preemption._replace(new_worker_pool=new_worker_pool)
        self._last_step_time = time
        self._state = TaskState.RUNNING

    def finish(self, time: int):
        """Completes the execution of the task at the given simulation time.

        If the remaining time is not 0, the task is considered to be preempted.

        Args:
            time (`int`): The simulation time (in us) at which the task was
                finished.
        """
        if self.state not in [TaskState.RUNNING, TaskState.PREEMPTED]:
            raise ValueError(f"Task {self.id} is not RUNNING or PREEMPTED right now.")
        self._completion_time = time
        if self._remaining_time == 0:
            self._state = TaskState.COMPLETED
        else:
            self._state = TaskState.EVICTED
        self._logger.debug(f"Finished execution of {self} at time {time}")
        # TODO (Sukrit): We should notify the `Job` of the completion of this
        # particular task, so it can release new tasks to the scheduler.

    def update_remaining_time(self, time: int):
        """Updates the remaining time of the task to simulate any runtime
        variabilities.

        Args:
            time (`int`): The new remaining time (in us) to update the task
                with.

        Raises:
            `ValueError` if the task is COMPLETED / EVICTED, or time < 0.
        """
        if self.is_complete():
            raise ValueError(
                f"The remaining time of COMPLETED/EVICTED "
                f"task {self.id} cannot be updated."
            )
        if time < 0:
            raise ValueError("Trying to set a negative value for " "remaining time.")
        self._remaining_time = time

    def update_deadline(self, new_deadline: int):
        """Updates the absolute deadline of the task to simulate any dynamic deadlines.

        Args:
            new_deadline (`int`): The new aboslute deadline (in us) to update the task
                with.

        Raises:
            `ValueError` if the new_deadline < 0.
        """
        self._deadline = new_deadline

    def is_complete(self) -> bool:
        """Check if the task has finished its execution.

        To return True, the task must be in either EVICTED / COMPLETED state.

        Returns:
            `True` if the task has finished, `False` otherwise.
        """
        return self.state == TaskState.EVICTED or self.state == TaskState.COMPLETED

    def is_ready_to_run(self) -> bool:
        """Check if the task can be placed.

        Returns:
            `True` if the task is ready to start, `False` otherwise.
        """
        return self.state == TaskState.RELEASED or self.state == TaskState.PREEMPTED

    def __str__(self):
        if self.state == TaskState.VIRTUAL:
            if self.release_time == -1:
                return (
                    f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state})"
                )
            else:
                if self.deadline == -1:
                    return (
                        f"Task(name={self.name}, id={self.id}, job={self.job}, "
                        f"timestamp={self.timestamp}, state={self.state}, "
                        f"release_time={self.release_time})"
                    )
                else:
                    return (
                        f"Task(name={self.name}, id={self.id}, job={self.job}, "
                        f"timestamp={self.timestamp}, state={self.state}, "
                        f"release_time={self.release_time}, deadline={self.deadline})"
                    )
        elif self.state == TaskState.RELEASED:
            return (
                f"Task(name={self.name}, id={self.id}, job={self.job}, "
                f"timestamp={self.timestamp}, state={self.state}, "
                f"release_time={self.release_time}, deadline={self.deadline})"
            )
        elif self.state == TaskState.RUNNING:
            return (
                f"Task(name={self.name}, id={self.id}, job={self.job}, "
                f"timestamp={self.timestamp}, state={self.state}, "
                f"start_time={self.start_time}, deadline={self.deadline}, "
                f"remaining_time={self.remaining_time}, "
                f"worker_pool={self.worker_pool_id})"
            )
        elif self.state == TaskState.PREEMPTED:
            return (
                f"Task(name={self.name}, id={self.id}, job={self.job}, "
                f"timestamp={self.timestamp}, state={self.state}, "
                f"preemption_time={self.preemption_time}, deadline={self.deadline}, "
                f"remaining_time={self.remaining_time}, "
                f"old_worker_pool={self.last_preemption.old_worker_pool})"
            )
        elif self.is_complete():
            return (
                f"Task(name={self.name}, id={self.id}, job={self.job}, "
                f"timestamp={self.timestamp}, state={self.state}, "
                f"completion_time={self.completion_time}, deadline={self.deadline})"
            )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return uuid.UUID(self.id) == uuid.UUID(other.id)

    def __lt__(self, other):
        if self == other:
            return False
        return self.deadline < other.deadline

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def resource_requirements(self):
        return self._resource_reqs

    @property
    def runtime(self):
        return self._expected_runtime

    @property
    def release_time(self):
        return self._release_time

    @property
    def intended_release_time(self):
        return self._intended_release_time

    @property
    def deadline(self):
        return self._deadline

    @property
    def job(self):
        return self._creating_job

    @property
    def state(self):
        return self._state

    @property
    def start_time(self):
        return self._start_time

    @property
    def preemption_time(self):
        return self.last_preemption.preemption_time if self.last_preemption else -1

    @property
    def last_preemption(self) -> Optional[Preemption]:
        return None if len(self._preemptions) == 0 else self._preemptions[-1]

    @property
    def remaining_time(self):
        return self._remaining_time

    @property
    def completion_time(self):
        return self._completion_time

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def worker_pool_id(self):
        return self._worker_pool_id


class TaskGraph(Graph[Task]):
    """A `TaskGraph` represents a directed graph of task dependencies that
    arise due to either the structure of the computation represented in the
    `JobGraph` or the timestamp ordering semantics employed by ERDOS.

    A `TaskGraph` is a runtime entity that constantly evolves as more tasks
    are released by the `JobGraph` and added to the given `TaskGraph`.
    """

    def __init__(self, tasks: Optional[Mapping[Task, Sequence[Task]]] = {}):
        super().__init__(tasks)

    def add_task(self, task: Task, children: Optional[Sequence[Task]] = []):
        """Adds the task to the graph along with the given children.

        Args:
            task (`Task`): The task to be added to the graph.
            children (`Sequence[Task]`): The children of the task, if any.
        """
        self.add_node(task, *children)

    def notify_task_completion(self, task: Task, finish_time: int) -> Sequence[Task]:
        """Notify the completion of the task.

        The caller must set the type of the task completion before invoking
        this method to ensure that the proper dependencies are unlocked.

        Args:
            task (`Task`): The task that has finished execution.
            finish_time (`int`): The time (in us) at which the task finished.

        Returns:
            The set of tasks released by the completion of this task.

        Raises:
            `ValueError` if an incomplete task is passed.
        """
        # Ensure that the task is actually complete.
        if not task.is_complete():
            raise ValueError(f"Cannot notify TaskGraph of an incomplete task: {task}")

        # Release any tasks that can be unlocked by the completion.
        released_tasks = []
        for child in self.get_children(task):
            if child.state != TaskState.VIRTUAL:
                # Task was already released when another of its parent tasks completed
                # simulatenously with the argument parent task given to this method.
                continue
            if all(map(lambda task: task.is_complete(), self.get_parents(child))):
                if child.release_time == -1:
                    # If the child does not have a release time, then set it to now,
                    # which is the time of the completion of the last parent task.
                    child.release(finish_time)
                else:
                    earliest_release = child.release_time
                    # Update the task's release time if parent tasks delayed it.
                    for parent in self.get_parents(child):
                        earliest_release = max(earliest_release, parent.completion_time)
                    child.release(earliest_release)
                released_tasks.append(child)
        return released_tasks

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
        time: int,
        lookahead: int = 0,
        preemption: bool = False,
        worker_pools: "WorkerPools" = None,  # noqa: F821
    ) -> Sequence[Task]:
        """Retrieves the all the tasks that are in RELEASED, PREEMPTED, or
        EVICTED state, and tasks that are expected to be released by time + lookahead.

        Returns:
            A list of tasks.
        """
        tasks = self.filter(
            lambda task: (
                task.state == TaskState.RELEASED
                and task.release_time <= time + lookahead
            )
            or task.state == TaskState.PREEMPTED
            or task.state == TaskState.EVICTED
        )

        # No need to add already running tasks if preemption is not enabled.
        if preemption:
            if worker_pools:
                # Adding the tasks placed on the given worker pool.
                tasks.extend(worker_pools.get_placed_tasks())
            else:
                # No worker pool provided. Getting all RUNNING tasks.
                tasks.extend(self.filter(lambda task: task.state == TaskState.RUNNING))

        task_queue = deque([])
        estimated_completion_time = {}
        # Estimate the completion time of materialized tasks.
        for task in self.get_nodes():
            if task.state == TaskState.COMPLETED:
                estimated_completion_time[task] = task.completion_time
            elif task.state in [
                TaskState.RUNNING,
                TaskState.RELEASED,
                TaskState.PREEMPTED,
                TaskState.EVICTED,
            ]:
                estimated_completion_time[task] = time + task.remaining_time
            else:
                continue
            task_queue.append(task)

        # Estimate the completion time of VIRTUAL tasks.
        while len(task_queue) > 0:
            task = task_queue.popleft()
            completion_time = estimated_completion_time[task]
            for child_task in self.get_children(task):
                if child_task.state != TaskState.VIRTUAL:
                    # Skip the task because we've already set its completion time.
                    continue
                child_completion_time = completion_time + child_task.remaining_time
                if child_task.release_time:
                    child_completion_time = max(
                        child_completion_time,
                        child_task.release_time + child_task.remaining_time,
                    )
                if (
                    child_task not in estimated_completion_time
                    or child_completion_time > estimated_completion_time[child_task]
                ):
                    estimated_completion_time[child_task] = child_completion_time
                    task_queue.append(child_task)

        # Add the tasks that are within the lookahead.
        tasks.extend(
            self.filter(
                lambda task: task.state == TaskState.VIRTUAL
                and task in estimated_completion_time
                and estimated_completion_time[task] < time + lookahead
            )
        )

        assert all(
            map(
                lambda task: task.release_time is None
                or task.release_time <= time + lookahead,
                tasks,
            )
        ), "Tasks send for scheduling beyond the scheduler lookahead"
        return tasks

    def release_tasks(self, time: Optional[int] = None) -> Sequence[Task]:
        """Releases the set of tasks that have no dependencies and are thus
        available to run.

        Args:
            time (`Optional[int]`): The simulation time (inus) at which to
                release the task. If None, the time should have been specified
                at task construction time.

        Returns:
            A list of tasks that can be run (are in RELEASED state).

        Raises:
            `ValueError` if no `time` for release is passed, and the tasks
            were not instantiated with a `release_time`.
        """
        tasks_to_be_released = []
        for task in self.get_nodes():
            parents = self.get_parents(task)
            if len(parents) == 0 or all(map(lambda task: task.is_complete(), parents)):
                tasks_to_be_released.append(task)

        # Release the tasks.
        for task in tasks_to_be_released:
            task.release(time)
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
            return TaskGraph(tasks=tasks)
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

    def get_source_tasks(self) -> Sequence[Task]:
        """Retrieve the source tasks from this instance of a TaskGraph.

        This method returns multiple instances of Tasks with different
        timestamps.

        Returns:
            A `Sequence[Task]` of tasks that have no dependencies on any
            tasks with the same timestamps.
        """
        return self.filter(self.is_source_task)

    def dilate(self, difference: int):
        """Dilate the time between occurrence of events of successive
        logical timestamps according to the given difference.

        If the provided difference is greater than the time between the
        occurrence of two events, the effect is a slowdown of the events.

        If the provided difference is smaller than the time between the
        occurrence of two events, the effect is a speedup of the events.

        This method changes the `TaskGraph` in-place by modifying the release
        time of the actual tasks.

        Args:
            difference (`int`): The time difference (in us) to keep between the
                occurrence of two source Tasks of different timestamps.
        """
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
            average_offset = int(sum(offsets) / len(offsets))
            for task in child_graph.get_nodes():
                if not child_graph.is_source_task(task):
                    task._release_time -= average_offset
                    task._deadline -= average_offset

    def merge(self, task_graphs: Sequence["TaskGraph"]) -> "TaskGraph":
        """Merge the given task_graphs after ordering them by timestamps.

        Args:
            task_graphs (`Sequence[TaskGraph]`): A sequence of task graphs to
                be merged.

        Returns:
            The merged TaskGraph from the sequence ordered by timestamp.
        """
        raise NotImplementedError("Merging of Taskgraphs has not been implemented yet.")

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
