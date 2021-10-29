import uuid
from enum import Enum
from collections import namedtuple, defaultdict
from typing import Mapping, Sequence, Optional

from workload import Job, Resources

Paused = namedtuple("Paused", "pause_time, restart_time")


class TaskState(Enum):
    """Represents the different states that a Task can potentially be in."""
    VIRTUAL = 1    # The Task is expected to be generated in the future.
    RELEASED = 2   # The Task has been released, and is waiting to be run.
    RUNNING = 3    # The Task has begun execution, and is currently running.
    PAUSED = 4     # The Task had begun execution but is currently paused.
    EVICTED = 5    # The Task has been evicted before completing.
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
        resource_requirement (`Resources`): The set of resources required by
            this task.
        release_time (`float`): The time at which the task was released by the
            job.
        runtime (`float`): The expected runtime of this task.
        deadline (`float`): The deadline by which the task should complete.
        state (`TaskState`): Defines the state of the task.
        start_time (`float`): The time at which the task was started (only
            available if state != TaskState.RELEASED, -1 otherwise)
        remaining_time (`float`): The time remaining to finish the completion
            of the task.
        completion_time (`float`): The time at which the task completed / was
            preempted (only available if state is either EVICTED / COMPLETED,
            -1 otherwise)
    """
    def __init__(self, name: str, job: Job, resource_requirement: Resources,
                 runtime: float, deadline: float):
        self._name = name
        self._creating_job = job
        self._resource_reqs = resource_requirement
        self._expected_runtime = runtime
        self._deadline = deadline
        self._id = uuid.uuid4()

        # The timestamps maintained for each state of the task.
        self._release_time = -1     # (VIRTUAL -> RELEASED)
        self._start_time = -1       # (RELEASED -> RUNNING)
        self._completion_time = -1  # (RUNNING -> EVICTED / COMPLETED)
        self._paused_times = []     # (RUNNING -> PAUSED)

        # The data required for managing the execution of a particular task.
        self._remaining_time = runtime
        self._state = TaskState.VIRTUAL

    def release(self, time: float):
        """Release the task and transition away from the virtual state.

        Args:
            time (`float`): The simulation time at which to release the task.
        """
        self._release_time = time
        self._state = TaskState.RELEASED

    def start(self, time: float):
        """Begins the execution of the task at the given simulator time.

        Args:
            time (`float`): The simulation time at which to begin the task.
        """
        self._start_time = time
        self._state = TaskState.RUNNING

    def step(self, step_size: float = 1):
        """Steps the task for the given `step_size` (default 1 time step).

        Args:
            step_size (`float`): The amount of time for which to step the task.
        """
        raise NotImplementedError("step() has not been implemented yet.")

    def pause(self, time: float):
        """Pauses the execution of the task at the given simulation time.

        Args:
            time (`float`): The simulation time at which to pause the task.
        """
        self._paused_times.append(Paused(time, -1))
        self._state = TaskState.PAUSED

    def resume(self, time: float):
        """Continues the execution of the task at the given simulation time.

        Args:
            time (`float`): The simulation time at which to restart the task.
        """
        self._paused_times[-1]._replace(restart_time=time)
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
        # TODO (Sukrit): We should notify the `Job` of the completion of this
        # particular task, so it can release new tasks to the scheduler.

    def update_runtime(self, new_runtime: float):
        """Updates the runtime of the task to simulate any runtime
        variabilities.

        Args:
            new_runtime (`float`): The new runtime to update the task with.
        """
        self._expected_runtime = new_runtime

    def update_deadline(self, new_deadline: float):
        """Updates the deadline of the task to simulate any dynamic deadlines.

        Args:
            new_deadline (`float`): The new deadline to update the task with.
        """
        self._deadline = new_deadline

    def is_complete(self) -> bool:
        """Check if the task has finished its execution.

        To return True, the task must be in either EVICTED / COMPLETED state.

        Returns:
            `True` if the task has finished, `False` otherwise.
        """
        return (self.state == TaskState.EVICTED or
                self.state == TaskState.COMPLETED)

    def __str__(self):
        return "Task(name={}, id={}, job={})".format(
                self.name, self.id, self.job)

    def repr(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

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
    def remaining_time(self):
        return self._remaining_time

    @property
    def completion_time(self):
        return self._completion_time


class TaskGraph(object):
    """A `TaskGraph` represents a directed graph of task dependencies that
    arise due to either the structure of the computation represented in the
    `JobGraph` or the timestamp ordering semantics employed by ERDOS.

    A `TaskGraph` is a runtime entity that constantly evolves as more tasks
    are released by the `JobGraph` and added to the given `TaskGraph`.
    """
    def __init__(self, tasks: Optional[Mapping[Task, Sequence[Task]]] = None):
        self._task_graph = defaultdict(list) if tasks is None else tasks

    def add_task(self, task: Task, children: Optional[Sequence[Task]] = None):
        """Adds the task to the graph along with the given children.

        Args:
            task (`Task`): The task to be added to the graph.
            children (`Sequence[Task]`): The children of the task, if any.
        """
        _children = [] if children is None else children

        # Add the parent task with the given children.
        if task in self._task_graph:
            print("The task {task} is already in the graph. Skipping.",
                  format(task=task))
        else:
            self._task_graph[task].extend(_children)

        # Add all the children tasks with an empty children list.
        for child in _children:
            self._task_graph[child].extend([])

    def notify_task_completion(self, task: Task) -> Sequence[Task]:
        """Notify the completion of the task.

        The caller must set the type of the task completion before invoking
        this method to ensure that the proper dependencies are unlocked.

        Returns:
            The set of tasks released by the completion of this task.
        """
        raise NotImplementedError("The notification of the completion of tasks\
            to the task graph has not been implemented yet.")

    def add_child(self, task: Task, child: Task):
        """Adds a child to the `Task` in the task graph.

        Args:
            task (`Task`): The task, to which the child needs to be added.
            child (`Task`): The child task to be added.
        """
        self._task_graph[task].append(child)
        self._task_graph[child].extend([])

    def find(self, task_id: uuid.UUID):
        """Finds the task with the given ID.

        Use this method to retrieve the instance of the task from the graph,
        and query / change its parameters.

        Args:
            task_id (`uuid.UUID`): Find a task with the given ID.
        """
        for task in self._task_graph:
            if task.id == task_id:
                return task

    def get_all_released_tasks(self) -> Sequence[Task]:
        """Retrieves the set of tasks that are available to run.

        Returns:
            A list of tasks that can be run (are in RELEASED state).
        """
        released_tasks = []
        for task in self._task_graph:
            if task.state == TaskState.RELEASED:
                released_tasks.append(task)
        return released_tasks

    def clean(self):
        """Cleans the `TaskGraph` of tasks that have finished completion.

        This is supposed to be called at regular intervals by the simulator so
        we don't end up a huge graph of finished tasks.

        We only clean up the tasks whose children have finished execution.
        """
        tasks_to_clean = []
        for task, children in self._task_graph:
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

        # Clean all the tasks.
        for task in tasks_to_clean:
            del self._task_graph[task]
