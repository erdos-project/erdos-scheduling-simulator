import logging
import random
import uuid
from copy import copy, deepcopy
from operator import attrgetter
from typing import List, Optional, Sequence, Tuple, Type

import utils
from workload import Resource, Resources, Task, TaskGraph, TaskState


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

    def __init__(
        self, name: str, resources: Resources, _logger: Optional[logging.Logger] = None
    ):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
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
        self._logger.debug(f"Placed {task} on {self}")

    def remove_task(self, task: Task):
        """Removes the task from this `Worker`.

        Args:
            task (`Task`): The task to be placed on this `Worker`.
        Raises:
            `ValueError` if the task was not placed on this worker.
        """
        if task not in self._placed_tasks:
            raise ValueError(f"The task {task} was not placed on {self.id} Worker.")
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

    def get_allocated_resources(self, task: Task) -> List[Tuple[Resource, float]]:
        """Retrieves the resources allocated to a given task from this Worker.

        Args:
            task: The task whose allocated resources need to be retrieved.

        Returns:
            A list of resource allocations whose each element is a (Resource,
            quantity allocated) pair.
        """
        return self._resources.get_allocated_resources(task)

    def step(self, current_time: int, step_size: int = 1) -> Sequence[Task]:
        """Steps all the tasks of this `Worker` by the given `step_size`.

        Args:
            current_time (`int`): The current time of the simulator (in us).
            step_size (`int`): The amount of time for which to step the
                tasks (in us).

        Returns:
            A set of tasks that have been completed.
        """
        completed_tasks = []
        # Invoke the step() method on all the tasks.
        for task in self._placed_tasks:
            if task.state != TaskState.RUNNING:
                self._logger.debug(f"Skipping stepping for task {task}.")
                continue
            self._logger.debug(
                f"Stepping through the execution of {task} for "
                f"{step_size} steps from time {current_time}."
            )
            if task.step(current_time, step_size):
                self._logger.debug(f"{task} finished execution on {self}.")
                completed_tasks.append(task)

        # Delete the completed tasks from the set of placed tasks.
        for task in completed_tasks:
            self.remove_task(task)
        return completed_tasks

    def __copy__(self):
        """A copy of the Worker uses the same ID, and copies the resource
        allocations of self.

        This is used by the schedulers to try scheduling decisions on a version
        of the current state of the Worker without affecting the state of the
        original Worker.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(
            instance,
            name=self.name,
            resources=copy(self.resources),
            _logger=self._logger,
        )
        instance._id = uuid.UUID(self.id)

        # Copy the placed tasks.
        for task, state in self._placed_tasks.items():
            instance._placed_tasks[task] = state

        return instance

    def __deepcopy__(self, memo):
        """A deepcopy of the Worker uses the same ID, and resets the resources
        to the initial state, thus undoing the effects of task placement.

        This is used by the schedulers to try scheduling decisions on the
        original state of the Worker without affecting the state of the
        original Worker.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(
            instance,
            name=self.name,
            resources=deepcopy(self.resources),
            _logger=self._logger,
        )
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
        return f"Worker(name={self.name}, id={self.id}, " f"resources={self.resources})"

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

    def __init__(
        self,
        name: str,
        workers: Optional[Sequence[Worker]] = [],
        scheduler: Optional[Type["BaseScheduler"]] = None,  # noqa
        _logger: Optional[logging.Logger] = None,
    ):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        self._name = name
        self._workers = {worker.id: worker for worker in workers}
        self._scheduler = scheduler
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._placed_tasks = {}  # Mapping[Task, str] from task to worker ID.

    def add_workers(self, workers: Sequence[Worker]):
        """Adds the given set of `Worker`s to this `WorkerPool`.

        Args:
            workers (`Sequence[Worker]`): The workers to be added.
        """
        for worker in workers:
            if worker.id in self._workers:
                self._logger.info(
                    f"Skipping addition of {worker} since it already "
                    f"exists in {self}"
                )
            else:
                self._logger.debug(f"Adding {worker} to {self}")
                self._workers[worker.id] = worker

    def place_task(self, task: Task) -> bool:
        """Places the task on this `WorkerPool`.

        The caller must ensure that the `WorkerPool` has enough resources to
        execute this task before invoking this method. Further, the
        `WorkerPool` is in charge of executing the task across its workers.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.

        Returns:
            False if the task could not be placed due to insufficient resources.
        """
        placement = None
        if self._scheduler is not None:
            # If a scheduler was provided, get a task placement from it.
            runtime_us, placement = self._scheduler.schedule(
                TaskGraph(tasks={task: []})[task], WorkerPools(self._workers)
            )
            # Add the runtime to the task start time.
            task._start_time += runtime_us
        else:
            # If there was no scheduler, find the first worker that can
            # accomodate the task given its resource requirements.
            for _id, _worker in self._workers.items():
                if _worker.can_accomodate_task(task):
                    placement = _id
                    break

        if placement is None:
            self._logger.warning(
                f"The {task} could not be placed on {self.id} WorkerPool."
            )
            return False
        else:
            self._workers[placement].place_task(task)
            self._placed_tasks[task] = placement
            return True

    def remove_task(self, task: Task):
        """Removes the task from this `WorkerPool`.

        Args:
            task (`Task`): The task to be placed on this `WorkerPool`.

        Raises:
            `ValueError` if the task was not placed on this worker pool.
        """
        if task not in self._placed_tasks:
            raise ValueError(f"The task {task} was not placed on {self.id} WorkerPool.")
        # Deallocate the resources and remove the placed task.
        self._workers[self._placed_tasks[task]].remove_task(task)
        del self._placed_tasks[task]

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s that are currently placed on this `WorkerPool`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        return list(self._placed_tasks.keys())

    def step(self, current_time: int, step_size: int = 1) -> Sequence[Task]:
        """Steps all the tasks of this `WorkerPool` by the given `step_size`.

        Args:
            current_time (`int`): The current time of the simulator  (in us).
            step_size (`int`): The amount of time for which to step the
                workers (in us).

        Returns:
            The set of tasks that have finished execution.
        """
        completed_tasks = []
        # Invoke the step() method on all the workers.
        for _, worker in self._workers.items():
            self._logger.debug(
                f"Stepping through the execution of {worker} for {step_size} "
                f"steps from time {current_time}"
            )
            completed_tasks.extend(worker.step(current_time, step_size))

        # Delete the completed tasks from the set of placed tasks.
        for task in completed_tasks:
            # We do not need to remove the task from the worker because it was
            # already removed while the worker stepped.
            del self._placed_tasks[task]
        return completed_tasks

    def can_accomodate_task(self, task: Task) -> bool:
        """Checks if any of the `Worker`s of this `WorkerPool` can accomodate
        the given `Task` based on its resource availability.

        Args:
            task (`Task`): The task to be placed on this `WorkerPool`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        return any(
            worker.can_accomodate_task(task) for worker in self._workers.values()
        )

    def get_utilization(self) -> Sequence[str]:
        """Retrieves the utilization of the resources of a particular WorkerPool in
        CSV format.

        The format of the output is:
        List["ResourceName,ResourceID,AllocatedQuantity,AvailableQuantity"]

        Returns:
            The utilization of the WorkerPool in CSV format.
        """
        # Add the resources of all the workers in this pool.
        final_resources = Resources(_logger=logging.getLogger("dummy"))
        for worker in self._workers.values():
            final_resources += worker.resources

        # Log the utilization from the final set of resources.
        resource_utilization = [
            ",".join(
                (
                    resource.name,
                    str(resource.id),
                    str(final_resources.get_allocated_quantity(resource)),
                    str(final_resources.get_available_quantity(resource)),
                )
            )
            for resource, _ in final_resources.resources
        ]
        return resource_utilization

    def get_allocated_resources(self, task: Task) -> List[Tuple[Resource, float]]:
        """Retrieves the resources allocated to a given task from this WorkerPool.

        Args:
            task: The task whose allocated resources need to be retrieved.

        Returns:
            A list of resource allocations whose each element is a (Resource,
            quantity allocated) pair.
        """
        return self._workers[self._placed_tasks[task]].get_allocated_resources(task)

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def workers(self):
        return list(self._workers.values())

    @property
    def resources(self):
        # Add the resources of all the workers in this pool.
        final_resources = Resources(_logger=self._logger)
        for worker in self._workers.values():
            final_resources += worker.resources
        return final_resources

    def __str__(self):
        return f"WorkerPool(name={self.name}, id={self.id})"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._workers)

    def __copy__(self):
        """A copy of the WorkerPool uses the same ID, and copies the state of
        the Workers, along with their current resource usage and placed tasks.

        This is used by the schedulers to try scheduling decisions on a version
        of the current state of the WorkerPool without affecting the state of
        the original WorkerPool.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(
            instance,
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
        """A deepcopy of the WorkerPool uses the same ID, and resets the
        Workers to the initial state, thus undoing the effects of the task
        placement.

        This is used by the schedulers to try scheduling decisions on the
        original state of the WorkerPool without affecting the state of the
        original WorkerPool.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(
            instance,
            name=self.name,
            workers=[deepcopy(w) for w in self._workers.values()],
            scheduler=copy(self._scheduler),
            _logger=self._logger,
        )
        instance._id = uuid.UUID(self.id)
        memo[id(self)] = instance
        return instance


class WorkerPools(object):
    """A collection of `WorkerPool`s."""

    def __init__(self, worker_pools: Sequence[WorkerPool]):
        self._wps = worker_pools

    def get_placed_tasks(self):
        placed_tasks = []
        for wp in self._wps:
            placed_tasks.extend(wp.get_placed_tasks())
        return placed_tasks

    def get_resource_ilp_encoding(self):
        """Constructs a map from resource name to (resource_start_index,
        resource_end_index) and a map from worker pool id to resource_id.

        The resource_start_index and resource_end_index are derived based on the
        available quantify of the given resource.
        """
        # Unique list of resource names -- not relying on set stability.
        resource_names = list(
            {r.name for wp in self._wps for r in wp.resources._resource_vector.keys()}
        )
        # Uniquify scrambles the order.
        resource_names.sort()

        res_type_to_index_range = {}
        res_index_to_wp_id = {}
        res_index_to_wp_index = {-1: -1}
        start_range_index = 0
        for res_name in resource_names:
            cur_range_index = start_range_index
            for index, wp in enumerate(self._wps):
                res_available = wp.resources.get_available_quantity(
                    Resource(name=res_name, _id="any")
                )
                for res_index in range(
                    cur_range_index, cur_range_index + res_available
                ):
                    res_index_to_wp_id[res_index] = wp.id
                    res_index_to_wp_index[res_index] = index
                cur_range_index += res_available
            res_type_to_index_range[res_name] = (start_range_index, cur_range_index)
            start_range_index = cur_range_index
        return (
            res_type_to_index_range,
            res_index_to_wp_id,
            res_index_to_wp_index,
            len(self._wps),
        )

    def __copy__(self):
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, [copy(wp) for wp in self._wps])
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, [deepcopy(wp) for wp in self._wps])
        return instance
