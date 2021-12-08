import uuid
import logging
from copy import copy, deepcopy
from typing import Optional, Sequence, Type

import utils
from workload import Resources, Task, TaskState


class Worker(object):
    """A `Worker` is a virtual abstraction over a single machine.

    A `Worker` "owns" a certain set of `Resource`s that execute a set of
    `Task`s assigned to it.

    Args:
        name (`str`): A name assigned to the particular instance of the Worker.
        resources (`Resource`): The set of `Resource`s owned by this worker.
        id (`UUID`): The ID of this particular Worker.
        _logger(`Optional[logging.Logger]`): The logger to use to log the
            results of the execution.
    """
    def __init__(self, name: str, resources: Resources,
                 _logger: Optional[logging.Logger] = None):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        self._name = name
        self._id = uuid.uuid4()
        self._resources = resources
        self._placed_tasks = {}  # Mapping[Task, TaskState]

    def place_task(self, task: Task):
        """Places the task on this `Worker`.

        The caller must check that the `Worker` can accomodate this task by
        invoking `can_accomodate_task`.

        Args:
            task (`Task`): The task to be placed on this `Worker`.
        """
        self._resources.allocate_multiple(task.resource_requirements, task)
        self._placed_tasks[task] = task.state
        self._logger.debug("Placed {} on {}".format(task, self))

    def preempt_task(self, task: Task):
        """Preempts the given `Task` and frees the resources.

        Args:
            task (`Task`): The task to be preempted from this `Worker`.

        Raises:
            `ValueError` if the task was not placed on this worker, or is not
            paused by the caller.
        """
        if task not in self._placed_tasks:
            raise ValueError("The task was not placed on this Worker.")
        if task.state != TaskState.PAUSED:
            raise ValueError("The task is not in PAUSED state.")

        # Deallocate the resources and remove the placed task.
        self._resources.deallocate(task)
        del self._placed_tasks[task]

    def can_accomodate_task(self, task: Task) -> bool:
        """Checks if this `Worker` can accomodate the given `Task` based on
        its resource availability.

        Args:
            task (`Task`): The task to be placed on this `Worker`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        return self._resources > task.resource_requirements

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task` that is currently placed on this `Worker`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        placed_tasks = []
        for task, _ in self._placed_tasks.items():
            placed_tasks.append(task)
        return placed_tasks

    def step(self, current_time: float, step_size: float = 1.0) -> \
            Sequence[Task]:
        """Steps all the tasks of this `Worker` by the given `step_size`.

        Args:
            current_time (`float`): The current time of the simulator loop.
            step_size (`float`): The amount of time for which to step the
                tasks.

        Returns:
            A set of tasks that have been completed.
        """
        completed_tasks = []
        # Invoke the step() method on all the tasks.
        for task in self._placed_tasks:
            self._logger.debug("Stepping through the execution of {} for {} "
                               "steps from time {}.".format(
                                   task, step_size, current_time))
            if task.step(current_time, step_size):
                self._logger.debug("{} finished execution on {}.".
                                   format(task, self))
                completed_tasks.append(task)
                # Resave the task as now completed / evicted.
                self._placed_tasks[task] = task.state
        return completed_tasks

    def __copy__(self):
        """ A copy of the Worker uses the same ID, and copies the resource
        allocations of self.

        This is used by the schedulers to try scheduling decisions on a version
        of the current state of the Worker without affecting the state of the
        original Worker.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance,
                     name=self.name,
                     resources=copy(self.resources),
                     _logger=self._logger)
        instance._id = uuid.UUID(self.id)

        # Copy the placed tasks.
        for task, state in self._placed_tasks.items():
            instance._placed_tasks[task] = state

        return instance

    def __deepcopy__(self, memo):
        """ A deepcopy of the Worker uses the same ID, and resets the resources
        to the initial state, thus undoing the effects of task placement.

        This is used by the schedulers to try scheduling decisions on the
        original state of the Worker without affecting the state of the
        original Worker.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance,
                     name=self.name,
                     resources=deepcopy(self.resources),
                     _logger=self._logger)
        instance._id = uuid.UUID(self.id)
        memo[id(self)] = instance
        return instance

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def resources(self):
        return self._resources

    def __str__(self):
        return "Worker(name={}, id={}, resources={})".\
               format(self.name, self.id, self.resources)

    def __repr__(self):
        return str(self)


class WorkerPool(object):
    """A `WorkerPool` acts as a collection of `Worker`s, and may perform its
    own scheduling (for a two-level scheduling architecture) across its
    workers.

    A Scheduler only works at the level of a WorkerPool, and a single Worker
    must be abstracted into a WorkerPool before it can be used by the
    Scheduler.

    However, to schedule a task across its `Worker`s, developers can provide
    a `Scheduler` instance that is in charge of scheduling the placed task.

    Args:
        name (`str`): A name assigned to this WorkerPool.
        id (`UUID`): The ID of the particular WorkerPool.
        workers (`Optional[Sequence[Worker]]`): The set of workers assigned to
            this WorkerPool, if any.
        scheduler (`Optional[Type[BaseScheduler]]`): The second-level scheduler
            implementation that schedules tasks assigned to this WorkerPool
            across its Workers.
        _logger (`Optional[absl.flags]`): The flags with which the app was
            initiated, if any.
    """
    def __init__(self, name: str, workers: Optional[Sequence[Worker]] = [],
                 scheduler: Optional[Type['BaseScheduler']] = None,
                 _logger: Optional[logging.Logger] = None):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        self._name = name
        self._workers = {worker.id: worker for worker in workers}
        self._scheduler = scheduler
        self._id = uuid.uuid4()
        self._placed_tasks = {}  # Mapping[Task, str] from task to worker ID.

    def add_workers(self, workers: Sequence[Worker]):
        """Adds the given set of `Worker`s to this `WorkerPool`.

        Args:
            workers (`Sequence[Worker]`): The workers to be added.
        """
        for worker in workers:
            if worker.id in self._workers:
                self._logger.info("Skipping addition of {} since it already "
                                  "exists in {}".format(worker, self))
            else:
                self._logger.debug("Adding {} to {}".format(worker, self))
                self._workers[worker.id] = worker

    def place_task(self, task: Task):
        """Places the task on this `WorkerPool`.

        The caller must ensure that the `WorkerPool` has enough resources to
        execute this task before invoking this method. Further, the
        `WorkerPool` is in charge of executing the task across its workers.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.

        Raises:
            `ValueError` if the task could not be placed due to insufficient
            resources.
        """
        placement = None
        if self._scheduler is not None:
            # If a scheduler was provided, get a task placement from it.
            runtime, placement = self._scheduler.schedule(
                                        [task],  # Only this task is available.
                                        None,    # No task graph.
                                        self._workers)
            # Add the runtime to the task start time.
            task._start_time += runtime
        else:
            # If there was no scheduler, find the first worker that can
            # accomodate the task given its resource requirements.
            for _id, _worker in self._workers.items():
                if _worker.can_accomodate_task(task):
                    placement = _id
                    break

        if placement is None:
            raise ValueError("The task ({}) could not be placed.".format(task))
        else:
            self._workers[placement].place_task(task)
            self._placed_tasks[task] = placement

    def preempt_task(self, task: Task):
        """Preempts the given `Task` and frees the resources.

        Args:
            task (`Task`): The task to be preempted from this `WorkerPool`.

        Raises:
            `ValueError` if the task was not placed on this pool, or is not
            paused by the caller.
        """
        # Find the worker where the task was placed, preempt it from there,
        # and remove it from this worker pool's placed tasks.
        self._workers[self._placed_tasks[task]].preempt_task(task)
        del self._placed_tasks[task]

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s that are currently placed on this `WorkerPool`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        return list(self._placed_tasks.keys())

    def step(self, current_time: float, step_size: float = 1.0) ->\
            Sequence[Task]:
        """Steps all the tasks of this `WorkerPool` by the given `step_size`.

        Args:
            current_time (`float`): The current time of the simulator loop.
            step_size (`float`): The amount of time for which to step the
                workers.

        Returns:
            The set of tasks that have finished execution.
        """
        completed_tasks = []
        # Invoke the step() method on all the workers.
        for _, worker in self._workers.items():
            self._logger.debug("Stepping through the execution of {} for {} "
                               "steps from time {}".format(
                                   worker, step_size, current_time))
            completed_tasks.extend(worker.step(current_time, step_size))
        return completed_tasks

    def can_accomodate_task(self, task: Task) -> bool:
        """Checks if any of the `Worker`s of this `WorkerPool` can accomodate
        the given `Task` based on its resource availability.

        Args:
            task (`Task`): The task to be placed on this `WorkerPool`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        return any(worker.can_accomodate_task(task)
                   for worker in self._workers.values())

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def workers(self):
        return list(self._workers.values())

    def __str__(self):
        return "WorkerPool(name={}, id={})".format(self.name, self.id)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._workers)

    def __copy__(self):
        """ A copy of the WorkerPool uses the same ID, and copies the state of
        the Workers, along with their current resource usage and placed tasks.

        This is used by the schedulers to try scheduling decisions on a version
        of the current state of the WorkerPool without affecting the state of
        the original WorkerPool.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance,
                     name=self.name,
                     workers=[copy(w) for w in self._workers.values()],
                     scheduler=copy(self._scheduler),
                     _logger=self._logger,
                     )
        instance._id = uuid.UUID(self.id)

        # Copy the placed tasks.
        for task, worker_id in self._placed_tasks.items():
            instance._placed_tasks[task] = worker_id

        return instance

    def __deepcopy__(self, memo):
        """ A deepcopy of the WorkerPool uses the same ID, and resets the
        Workers to the initial state, thus undoing the effects of the task
        placement.

        This is used by the schedulers to try scheduling decisions on the
        original state of the WorkerPool without affecting the state of the
        original WorkerPool.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance,
                     name=self.name,
                     workers=[deepcopy(w) for w in self._workers.values()],
                     scheduler=copy(self._scheduler),
                     _logger=self._logger,
                     )
        instance._id = uuid.UUID(self.id)
        memo[id(self)] = instance
        return instance
