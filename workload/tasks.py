import uuid
import logging
from enum import Enum
from collections import namedtuple, defaultdict
from typing import Mapping, Sequence, Optional, Union

import utils
from workload import Job, Resources

Paused = namedtuple("Paused", "pause_time, restart_time")


class TaskState(Enum):
    """Represents the different states that a Task can potentially be in."""
    VIRTUAL = 1  # The Task is expected to be generated in the future.
    RELEASED = 2  # The Task has been released, and is waiting to be run.
    RUNNING = 3  # The Task has begun execution, and is currently running.
    PAUSED = 4  # The Task had begun execution but is currently paused.
    EVICTED = 5  # The Task has been evicted before completing.
    COMPLETED = 6  # The Task has successfully completed.


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
        release_time (`float`): The time at which the task was released by the
            job.
        runtime (`float`): The expected runtime of this task.
        deadline (`float`): The deadline by which the task should complete.
        state (`TaskState`): Defines the state of the task.
        timestamp (`int`): The timestamp for the Task (single dimension).
        start_time (`float`): The time at which the task was started (only
            available if state != TaskState.RELEASED, -1 otherwise)
        remaining_time (`float`): The time remaining to finish the completion
            of the task.
        completion_time (`float`): The time at which the task completed / was
            preempted (only available if state is either EVICTED / COMPLETED,
            -1 otherwise)
        _logger(`Optional[logging.Logger]`): The logger to use to log the
            results of the execution.
    """

    def __init__(self,
                 name: str,
                 job: Job,
                 resource_requirements: Resources,
                 runtime: float,
                 deadline: float,
                 timestamp: int = None,
                 release_time: Optional[float] = -1,
                 start_time: Optional[float] = -1,
                 completion_time: Optional[float] = -1,
                 _logger: Optional[logging.Logger] = None):
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
        self._id = uuid.uuid4()

        # The timestamps maintained for each state of the task.
        # (VIRTUAL -> RELEASED)
        self._release_time = release_time
        # (RELEASED -> RUNNING)
        self._start_time = start_time
        # (RUNNING -> EVICTED / COMPLETED)
        self._completion_time = completion_time
        # (RUNNING -> PAUSED)
        self._paused_times = []

        # The data required for managing the execution of a particular task.
        self._remaining_time = runtime
        self._last_step_time = -1  # Time when this task was stepped through.
        self._state = TaskState.VIRTUAL

    def release(self, time: Optional[float] = None):
        """Release the task and transition away from the virtual state.

        Args:
            time (`Optional[float]`): The simulation time at which to release
                the task. If None, should be specified at task construction.
        """
        self._logger.debug(f"Transitioning {self} to {TaskState.RELEASED}")
        if time is None and self._release_time == -1:
            raise ValueError("Release time should be specified either while "
                             "creating the Task or when releasing it.")
        self._release_time = time if time is not None else self._release_time
        self._state = TaskState.RELEASED

    def start(self,
              time: Optional[float] = None,
              variance: Optional[float] = 0.0):
        """Begins the execution of the task at the given simulator time.

        Args:
            time (`Optional[float]`): The simulation time at which to begin the
                task. If None, should be specified at task construction.
            variance (`Optional[float]`): The percentage variation to add to
                the runtime of the task.

        Raises:
            `ValueError` if Task is not in `RELEASED`/`PAUSED` state yet.
        """
        if self.state not in (TaskState.RELEASED, TaskState.PAUSED):
            raise ValueError("Only RELEASED or PAUSED tasks can be started.")
        if time is None and self._start_time == -1:
            raise ValueError("Start time should be specified either while "
                             "creating the Task or when starting it.")

        remaining_time = utils.fuzz_time(self._remaining_time, variance)
        self._logger.debug(
            f"Transitioning {self} to {TaskState.RUNNING} at time {time} "
            f"with the remaining time {remaining_time}")
        self._start_time = time if time is not None else self._start_time
        self._last_step_time = time
        self._state = TaskState.RUNNING
        self.update_remaining_time(remaining_time)

    def step(self, current_time: float, step_size: float = 1) -> bool:
        """Steps the task for the given `step_size` (default 1 time step).

        Args:
            current_time (`float`): The current time of the simulator loop.
            step_size (`float`): The amount of time for which to step the task.

        Returns:
            `True` if the task has finished execution, `False` otherwise.
        """
        if (self.state != TaskState.RUNNING
                or self.start_time > current_time + step_size):
            # We cannot step a Task that's not supposed to be running.
            self._logger.warning(
                f"Cannot step {self} with start time {self.start_time} at "
                f"time {current_time} since it's either not RUNNING or isn't "
                f"supposed to start yet.")
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
                f"Remaining execution time: {self._remaining_time}")
            return False

    def pause(self, time: float):
        """Pauses the execution of the task at the given simulation time.

        Args:
            time (`float`): The simulation time at which to pause the task.

        Raises:
            `ValueError` if task is not RUNNING.
        """
        if self.state != TaskState.RUNNING:
            raise ValueError("Task is not RUNNING right now.")
        self._logger.debug(
            f"Transitioning {self} to {TaskState.PAUSED} at time {time}")
        self._paused_times.append(Paused(time, -1))
        self._state = TaskState.PAUSED

    def resume(self, time: float):
        """Continues the execution of the task at the given simulation time.

        Args:
            time (`float`): The simulation time at which to restart the task.

        Raises:
            `ValueError` if task is not PAUSED.
        """
        if self.state != TaskState.PAUSED:
            raise ValueError("Task is not PAUSED right now.")
        self._logger.debug(
            f"Transitioning {self} which was PAUSED at "
            f"{self._paused_times[-1].pause_time} to {TaskState.RUNNING} at "
            f"time {time}")
        self._paused_times[-1]._replace(restart_time=time)
        self._last_step_time = time
        self._state = TaskState.RUNNING

    def finish(self, time: float):
        """Completes the execution of the task at the given simulation time.

        If the remaining time is not 0, the task is considered to be preempted.

        Args:
            time (`float`): The simulation time at which the task was finished.
        """
        self._completion_time = time
        if self._remaining_time == 0:
            self._state = TaskState.COMPLETED
        else:
            self._state = TaskState.EVICTED
        self._logger.debug(f"Finished execution of {self} at time {time}")
        # TODO (Sukrit): We should notify the `Job` of the completion of this
        # particular task, so it can release new tasks to the scheduler.

    def update_remaining_time(self, time: float):
        """Updates the remaining time of the task to simulate any runtime
        variabilities.

        Args:
            time (`float`): The new remaining time to update the task with.

        Raises:
            `ValueError` if the task is COMPLETED / EVICTED, or time < 0.
        """
        if self.is_complete():
            raise ValueError("The remaining time of COMPLETED/EVICTED "
                             "tasks cannot be updated.")
        if time < 0:
            raise ValueError("Trying to set a negative value for "
                             "remaining time.")
        self._remaining_time = time

    def update_deadline(self, new_deadline: float):
        """Updates the deadline of the task to simulate any dynamic deadlines.

        Args:
            new_deadline (`float`): The new deadline to update the task with.

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
        return (self.state == TaskState.EVICTED
                or self.state == TaskState.COMPLETED)

    def __str__(self):
        if self.state == TaskState.VIRTUAL:
            if self.release_time == -1:
                return (
                    f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state})")
            else:
                return (
                    f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state}, "
                    f"release_time={self.release_time})")
        elif self.state == TaskState.RELEASED:
            return (f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state}, "
                    f"release_time={self.release_time})")
        elif self.state == TaskState.RUNNING:
            return (f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state}, "
                    f"start_time={self.start_time}, "
                    f"remaining_time={self.remaining_time})")
        elif self.state == TaskState.PAUSED:
            return (f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state}, "
                    f"pause_time={self.pause_time}, "
                    f"remaining_time={self.remaining_time})")
        elif self.is_complete():
            return (f"Task(name={self.name}, id={self.id}, job={self.job}, "
                    f"timestamp={self.timestamp}, state={self.state}, "
                    f"completion_time={self.completion_time})")

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return uuid.UUID(self.id) == uuid.UUID(other.id)

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
    def pause_time(self):
        return self._paused_times[-1].pause_time

    @property
    def remaining_time(self):
        return self._remaining_time

    @property
    def completion_time(self):
        return self._completion_time

    @property
    def timestamp(self):
        return self._timestamp


class TaskGraph(object):
    """A `TaskGraph` represents a directed graph of task dependencies that
    arise due to either the structure of the computation represented in the
    `JobGraph` or the timestamp ordering semantics employed by ERDOS.

    A `TaskGraph` is a runtime entity that constantly evolves as more tasks
    are released by the `JobGraph` and added to the given `TaskGraph`.
    """

    def __init__(self, tasks: Optional[Mapping[Task, Sequence[Task]]] = {}):
        self._task_graph = defaultdict(list)
        self.__parent_task_graph = defaultdict(list)
        self._max_timestamp = float('-inf')

        # Maintain the child and parent abstractions from the given taskset.
        for task, children in tasks.items():
            self.add_task(task, children)

    def add_task(self, task: Task, _children: Optional[Sequence[Task]] = []):
        """Adds the task to the graph along with the given children.

        Args:
            task (`Task`): The task to be added to the graph.
            children (`Sequence[Task]`): The children of the task, if any.
        """
        if len(_children) == 0:
            # If no children are provided, add just the task.
            if task.timestamp > self._max_timestamp:
                self._max_timestamp = task.timestamp
            self._task_graph[task].extend([])
        else:
            # Add the children to the graph individually.
            for child in _children:
                self.add_child(task, child)

    def notify_task_completion(self, task: Task, finish_time: float) ->\
            Sequence[Task]:
        """Notify the completion of the task.

        The caller must set the type of the task completion before invoking
        this method to ensure that the proper dependencies are unlocked.

        Args:
            task (`Task`): The task that has finished execution.
            finish_time (`float`): The time at which the task finished.

        Returns:
            The set of tasks released by the completion of this task.

        Raises:
            `ValueError` if an incomplete task is passed.
        """
        # Ensure that the task is actually complete.
        if not task.is_complete():
            raise ValueError(
                f"Cannot notify TaskGraph of an incomplete task: {task}")

        # Release any tasks that can be unlocked by the completion.
        released_tasks = []
        for child in self.get_children(task):
            if all(
                    map(lambda task: task.is_complete(),
                        self.get_parents(child))):
                if child.release_time == -1:
                    child.release(finish_time)
                else:
                    child.release()
                released_tasks.append(child)
        return released_tasks

    def add_child(self, task: Task, child: Task):
        """Adds a child to the `Task` in the task graph.

        Args:
            task (`Task`): The task, to which the child needs to be added.
            child (`Task`): The child task to be added.
        """
        # Maintain the maximum timestamp encountered in tasks.
        if task.timestamp > self._max_timestamp:
            self._max_timestamp = task.timestamp

        if child.timestamp > self._max_timestamp:
            self._max_timestamp = child.timestamp

        # Maintian the child and parent connections for the task.
        self._task_graph[task].append(child)
        self._task_graph[child].extend([])
        self.__parent_task_graph[child].append(task)

    def find(self, task_name: str) -> Sequence[Task]:
        """Find all the instances of a task with the given name.

        Use this method to retrieve the instances of a task from the graph,
        and query / change their parameters.

        Args:
            task_name (`str`): Find a task with the given name.

        Returns:
            A possibly empty `Sequence[Task]` with the given name.
        """
        return list(
            filter(lambda task: task.name == task_name,
                   self._task_graph.keys()))

    def get_children(self, task: Task) -> Sequence[Task]:
        """Retrieves the children of the given task.

        Args:
            task (`Task`): The task to retrieve the children of.

        Returns:
            The children of the given task.
        """
        if task not in self._task_graph:
            return []
        else:
            return self._task_graph[task]

    def get_parents(self, task: Task) -> Sequence[Task]:
        """Retrieves the parents of the given task.

        Args:
            task (`Task`): The task to retrieve the parents of.

        Returns:
            The parents of the given task.
        """
        if task not in self._task_graph:
            return []
        else:
            return self.__parent_task_graph[task]

    def get_released_tasks(self) -> Sequence[Task]:
        """Retrieves the set of tasks that are available to run.

        Returns:
            A list of tasks that can be run (are in RELEASED state).
        """
        released_tasks = []
        for task in self._task_graph:
            if task.state == TaskState.RELEASED:
                released_tasks.append(task)
        return released_tasks

    def release_tasks(self, time: Optional[float] = None) -> Sequence[Task]:
        """Releases the set of tasks that have no dependencies and are thus
        available to run.

        Args:
            time (`Optional[float]`): The simulation time at which to release
                the task. If None, the time should have been specified at task
                construction time.

        Returns:
            A list of tasks that can be run (are in RELEASED state).

        Raises:
            `ValueError` if no `time` for release is passed, and the tasks
            were not instantiated with a `release_time`.
        """
        tasks_to_be_released = []
        for task in self._task_graph:
            if len(self.__parent_task_graph[task]) == 0 or\
               all(map(lambda task: task.is_complete(),
                       self.__parent_task_graph[task])):
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
        for task, children in self._task_graph.items():
            # Has this task finished execution?
            if task.is_complete():
                # Check if all children have finished execution too.
                can_be_cleaned = True
                for child in children:
                    if not child.is_completed():
                        can_be_cleaned = False
                        break

                # If the task can be cleaned, add the entry to be removed.
                if can_be_cleaned:
                    tasks_to_clean.append(task)

        # Remove the task from the parent graph of all its children, and then
        # remove the task from the graph itself.
        for task in tasks_to_clean:
            for child in self.get_children(task):
                self.__parent_task_graph[child].remove(task)
            del self._task_graph[task]

    def __getitem__(self, slice_obj) ->\
            Union['TaskGraph', Sequence['TaskGraph']]:
        """Retrieve a slice of the TaskGraph as specified by the `slice_obj`.

        The `slice_obj` should specify the timestamps for which the slice is
        required. In case a slice object is passed, this method returns a
        `Sequence[TaskGraph]` of sliced TaskGraphs with the corresponding
        timestamps.
        """
        if isinstance(slice_obj, int):
            # Get the slice for a single timestamp.
            tasks = {}
            for task, children in self._task_graph.items():
                if task.timestamp == slice_obj:
                    # Maintain the task dependencies with the same timestamps.
                    same_timestamp_children = []
                    for child in children:
                        if child.timestamp == slice_obj:
                            same_timestamp_children.append(child)

                    # Add the task to the representation.
                    tasks[task] = same_timestamp_children
            return TaskGraph(tasks=tasks)
        elif isinstance(slice_obj, slice):
            return [
                self[index]
                for index in range(*slice_obj.indices(self._max_timestamp + 1))
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
        parents = self.__parent_task_graph[task]
        return (len(parents) == 0
                or (len(parents) == 1 and parents[0].name == task.name
                    and parents[0].timestamp == task.timestamp - 1))

    def get_source_tasks(self) -> Sequence[Task]:
        """Retrieve the source tasks from this instance of a TaskGraph.

        This method returns multiple instances of Tasks with different
        timestamps.

        Returns:
            A `Sequence[Task]` of tasks that have no dependencies on any
            tasks with the same timestamps.
        """
        return list(filter(self.is_source_task, self._task_graph.keys()))

    def dilate(self, difference: float):
        """Dilate the time between occurrence of events of successive
        logical timestamps according to the given difference.

        If the provided difference is greater than the time between the
        occurrence of two events, the effect is a slowdown of the events.

        If the provided difference is smaller than the time between the
        occurrence of two events, the effect is a speedup of the events.

        This method changes the `TaskGraph` in-place by modifying the release
        time of the actual tasks.

        Args:
            difference (`float`): The time difference to keep between the
                occurrence of two source Tasks of different timestamps.
        """
        for parent_graph, child_graph in zip(self[0:], self[1:]):
            # Fix the offsets of the source tasks according to the release
            # time of the parents.
            child_source_tasks = child_graph.get_source_tasks()
            offsets = []
            for child_source_task in child_source_tasks:
                parent_source_task = parent_graph.find(child_source_task.name)
                assert len(parent_source_task) == 1,\
                    f"Expected a single parent for: {child_source_task}"
                parent_source_task = parent_source_task[0]

                # Calculate the offset and set the release time according to
                # the parent task's release time and the provided difference.
                offset = (child_source_task.release_time -
                          (parent_source_task.release_time + difference))
                offsets.append(offset)
                child_source_task._release_time = (
                    parent_source_task.release_time + difference)

            # Calculate the average of the offsets of the source tasks and
            # offset the remainder of the tasks by the average.
            average_offset = sum(offsets) / len(offsets)
            for task in child_graph._task_graph:
                if not child_graph.is_source_task(task):
                    task._release_time -= average_offset

    def merge(self, task_graphs: Sequence['TaskGraph']) -> 'TaskGraph':
        """Merge the given task_graphs after ordering them by timestamps.

        Args:
            task_graphs (`Sequence[TaskGraph]`): A sequence of task graphs to
                be merged.

        Returns:
            The merged TaskGraph from the sequence ordered by timestamp.
        """
        raise NotImplementedError("Merging of Taskgraphs has not been "
                                  "implemented yet.")

    def __len__(self):
        return len(self._task_graph)

    def __str__(self):
        constructed_string = ""
        for task, children in self._task_graph.items():
            constructed_string += "{}_{}: {}\n".format(
                task.name, task.timestamp,
                list(
                    map(lambda t: "{}_{}".format(t.name, t.timestamp),
                        children)))
        return constructed_string
