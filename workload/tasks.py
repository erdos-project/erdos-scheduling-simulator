import uuid
from enum import Enum
from jobs import Job
from resources import Resources
from collections import namedtuple

Paused = namedtuple("Paused", "pause_time, restart_time")


class TaskState(Enum):
    """Represents the different states that a Task can potentially be in."""
    RELEASED = 1   # The Task has been released, and is waiting to be run.
    RUNNING = 2    # The Task has begun execution, and is currently running.
    PAUSED = 3     # The Task had begun execution but is currently paused.
    PREEMPTED = 4  # The Task has been preempted before completing.
    COMPLETED = 5  # The Task has successfully completed.


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
            preempted (only available if state is either PREEMPTED / COMPLETED,
            -1 otherwise)
    """
    def __init__(self, name: str, job: Job, resource_requirement: Resources,
                 release_time: float, runtime: float, deadline: float):
        self._name = name
        self._creating_job = job
        self._resource_reqs = resource_requirement
        self._release_time = release_time
        self._expected_runtime = runtime
        self._deadline = deadline
        self._id = uuid.uuid1()

        # The data required for managing the execution of a particular task.
        self._start_time, self._completion_time = -1, -1
        self._remaining_time = runtime
        self._state = TaskState.RELEASED
        self._paused_times = []  # A list of tuples of the type Paused.

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
            self._state = TaskState.PREEMPTED
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

    def __str__(self):
        return "Task(name={}, id={}, job={})".format(
                self.name, self.id, self.job)

    def repr(self):
        return str(self)

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
    def __init__(self):
        pass
