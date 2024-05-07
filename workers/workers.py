import logging
import random
import uuid
from copy import copy, deepcopy
from typing import List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

from utils import EventTime, setup_logging
from workload import (
    BatchStrategy,
    ExecutionStrategies,
    ExecutionStrategy,
    Resource,
    Resources,
    Task,
    TaskGraph,
    TaskState,
    Workload,
    WorkProfile,
)


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
            self._logger = setup_logging(name=self.__class__.__name__)

        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._resources = resources
        self._placed_tasks: Mapping[Task, ExecutionStrategy] = {}
        self._placed_batches: Mapping[BatchStrategy, Set[Task]] = {}
        self._batch_tasks_for_strategy: Mapping[BatchStrategy, Task] = {}
        self._available_profiles: Mapping[WorkProfile, ExecutionStrategy] = {}
        self._pending_profiles: Mapping[WorkProfile, ExecutionStrategy] = {}

    def load_profile(
        self, profile: WorkProfile, loading_strategy: ExecutionStrategy
    ) -> None:
        """Loads a given `WorkProfile` into the `Worker`. This method is used for
        loading models into the `Worker`, which a `Task` can then utilize to execute
        using an execution strategy.

        The caller must check that the `Worker` can accomodate this strategy by
        invoking `can_accomodate_strategy`. If there are not enough resources available,
        the method raises a `ValueError`.

        Args:
            profile (`WorkProfile`): The profile that represents the computation that
                is to be loaded into this `Worker`.
            loading_strategy (`ExecutionStrategy`): The strategy to be used to load the
                computation into this `Worker`.

        Raises:
            A `ValueError` if not enough resources are available to accomodate the
            loading strategy.
        """
        self._resources.allocate_multiple(loading_strategy.resources, profile)
        self._pending_profiles[profile] = copy(loading_strategy)
        self._logger.debug(
            f"Added the profile {profile} with the loading strategy "
            f"{loading_strategy} to the set of pending profiles."
        )

    def place_task(
        self, task: Task, execution_strategy: Union[ExecutionStrategy, BatchStrategy]
    ) -> None:
        """Places the task on this `Worker` using the given `execution_strategy`.

        The caller must check that the `Worker` can accomodate this strategy by
        invoking `can_accomodate_strategy`. If an instance of type `BatchStrategy` is
        provided to the method, the method will ensure that all `Task`s in the batch
        are accomodated together without requiring separate resources for each of them.

        Args:
            task (`Task`): The task to be placed on this `Worker`.
            execution_strategy (`Union[ExecutionStrategy, BatchStrategy]`): The
                strategy to be used for executing the task.
        """
        if isinstance(execution_strategy, BatchStrategy):
            # If the execution strategy is a batch strategy, then we need to allocate
            # the resources only if the batch is not already placed on this worker.
            if execution_strategy not in self._placed_batches:
                if execution_strategy.batch_size < 1:
                    raise ValueError(
                        f"The batch size {execution_strategy.batch_size} is invalid."
                    )
                # Create a virtual task that acts as a placeholder for the Resources.
                batch_task = Task(
                    name=f"BatchFor{task.name}",
                    task_graph=task.task_graph,
                    job=task.job,
                    deadline=task.deadline,
                )
                self._resources.allocate_multiple(
                    execution_strategy.resources, batch_task
                )

                # Add the task to the set of placed tasks for the given batch.
                self._placed_batches[execution_strategy] = set()
                self._placed_batches[execution_strategy].add(task)
                self._placed_tasks[task] = execution_strategy

                # Log the virtual task for the batch.
                self._batch_tasks_for_strategy[execution_strategy] = batch_task
                self._logger.debug(
                    f"Placed {task} on {self} as part of a new "
                    f"batch with the ID: {execution_strategy.id}."
                )
            else:
                if (
                    len(self._placed_batches[execution_strategy]) + 1
                    > execution_strategy.batch_size
                ):
                    raise RuntimeError(
                        f"Total of {len(self._placed_batches[execution_strategy]) + 1} "
                        f"tasks were placed in the batch, but the batch size is "
                        f"{execution_strategy.batch_size}."
                    )
                self._placed_batches[execution_strategy].add(task)
                self._placed_tasks[task] = execution_strategy
                self._logger.debug(
                    f"Placed {task} on {self} as part of an already "
                    f"placed batch with the ID: {execution_strategy.id}."
                )
        else:
            self._resources.allocate_multiple(execution_strategy.resources, task)
            self._placed_tasks[task] = execution_strategy
            self._logger.debug(
                f"Placed {task} on {self} with the "
                f"execution strategy {execution_strategy}."
            )

    def evict_profile(self, profile: WorkProfile) -> None:
        """Evicts the given `profile` and frees up the resources corresponding to the
        loading strategy used for the profile.

        Args:
            profile (`WorkProfile`): The profile that needs to be evicted from the
                set of available profiles at this `Worker`.

        Raises:
            A `ValueError` if the profile was not placed on this `Worker`.
        """
        if (
            profile not in self._available_profiles
            and profile not in self._pending_profiles
        ):
            raise ValueError(
                f"The profile {profile} was not available at Worker {self}."
            )

        # Deallocates the resources corresponding to the loading strategy and remove
        # the profile from the set of available profiles.
        self._resources.deallocate(profile)
        if profile in self._available_profiles:
            del self._available_profiles[profile]
        else:
            del self._pending_profiles[profile]

    def remove_task(self, current_time: EventTime, task: Task):
        """Removes the task from this `Worker`.

        If the `task` was executed as part of a `BatchStrategy`, then the method only
        deallocates the resources for the `Task` once the last `Task` in the batch has
        been requested to be removed.

        Args:
            current_time (`EventTime`): The time at which the removal of the `Task`
                was requested.
            task (`Task`): The task to be placed on this `Worker`.
        Raises:
            `ValueError` if the task was not placed on this worker.
        """
        if task not in self._placed_tasks:
            raise ValueError(f"The task {task} was not placed on Worker {self}.")

        execution_strategy = self._placed_tasks[task]
        if isinstance(execution_strategy, BatchStrategy):
            # If the Task was executed as part of a `Batch`, then we need to check if
            # this is the last task in the batch. If it is, then we need to deallocate
            # the resources for the batch. Otherwise, we just remove the task from the
            # set of placed tasks for the batch.
            remaining_tasks_in_batch = self._placed_batches.get(execution_strategy)
            if remaining_tasks_in_batch is None or task not in remaining_tasks_in_batch:
                raise RuntimeError(
                    f"The Task {task} was executed as part of a Batch "
                    f"{execution_strategy.id}, but was not found in the set of "
                    f"remaining tasks."
                )
            remaining_tasks_in_batch.remove(task)
            if len(remaining_tasks_in_batch) == 0:
                # The last Task in the batch was removed, so we need to deallocate the
                # resources for the batch.
                batch_task = self._batch_tasks_for_strategy.get(execution_strategy)
                if batch_task is None:
                    raise RuntimeError(
                        f"The Task {task} was executed as part of a Batch "
                        f"{execution_strategy.id}, but the corresponding batch "
                        f"task was not found."
                    )
                self._resources.deallocate(batch_task)
                del self._placed_batches[execution_strategy]
                del self._batch_tasks_for_strategy[execution_strategy]
                self._logger.debug(
                    "[%d] The last Task %s from the Batch %s was removed from the "
                    "Worker %s, and the batch was removed.",
                    current_time.to(EventTime.Unit.US).time,
                    task,
                    execution_strategy.id,
                    self,
                )
            self._placed_batches[execution_strategy] = remaining_tasks_in_batch
            del self._placed_tasks[task]
        else:
            # Deallocate the resources and remove the placed task.
            self._resources.deallocate(task)
            del self._placed_tasks[task]
            self._logger.debug(
                "[%d] The Task %s was removed from the Worker %s.",
                current_time.to(EventTime.Unit.US).time,
                task,
                self,
            )

    def can_accomodate_strategy(
        self, strategy: Union[ExecutionStrategy, BatchStrategy]
    ) -> bool:
        """Checks if this `Worker` can accomodate the given `ExecutionStrategy` based
        on the available resources. If the strategy is of type `BatchStrategy`, then the
        `Worker` can accomodate the strategy if another `Task` from the same batch was
        already placed.

        Args:
            strategy (`ExecutionStrategy`): The strategy that the `Worker` has to
                accomodate.

        Returns:
            `True` if the `Worker` has enough resources to execute this strategy, and
            `False` otherwise.
        """
        return self._resources > strategy.resources or (
            isinstance(strategy, BatchStrategy) and strategy in self._placed_batches
        )

    def get_compatible_strategies(
        self, execution_strategies: ExecutionStrategies
    ) -> ExecutionStrategies:
        """Get the set of execution strategies from the given strategies that are
        compatible with this `Worker` i.e., can be executed by this `Worker`.

        Args:
            execution_strategies (`ExecutionStrategies`): The initial set of available
                strategies that need to be filtered to the set of strategies executable
                by this `Worker`.

        Returns
            A new (possibly empty) `ExecutionStrategies` object that contains the set
            of strategies that can be executed by this `Worker`.
        """
        filtered_strategies = ExecutionStrategies()
        for strategy in execution_strategies:
            if self.can_accomodate_strategy(strategy):
                filtered_strategies.add_strategy(strategy)
        return filtered_strategies

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task` that is currently placed on this `Worker`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        return list(self._placed_tasks.keys())

    def get_allocated_resources(self, task: Task) -> List[Tuple[Resource, float]]:
        """Retrieves the resources allocated to a given task from this Worker.

        Args:
            task: The task whose allocated resources need to be retrieved.

        Returns:
            A list of resource allocations whose each element is a (Resource,
            quantity allocated) pair.
        """
        if task not in self._placed_tasks:
            raise RuntimeError(
                f"The task {task.unique_name} was not placed on {self.name}."
            )
        execution_strategy = self._placed_tasks[task]
        if isinstance(execution_strategy, BatchStrategy):
            # If the Task was executed as part of a `Batch`, then we need to check
            # for the resource allocations of the virtual batch task.
            if execution_strategy not in self._batch_tasks_for_strategy:
                raise RuntimeError(
                    f"The virtual batch task corresponding to the strategy "
                    f"{execution_strategy} was not found."
                )
            virtual_batch_task = self._batch_tasks_for_strategy[execution_strategy]
            return self._resources.get_allocated_resources(virtual_batch_task)
        return self._resources.get_allocated_resources(task)

    def step(
        self,
        current_time: EventTime,
        step_size: EventTime = EventTime(1, EventTime.Unit.US),
    ) -> Sequence[Task]:
        """Steps all the tasks of this `Worker` by the given `step_size`.

        Args:
            current_time (`int`): The current time of the simulator (in us).
            step_size (`int`): The amount of time for which to step the
                tasks (in us).

        Returns:
            A set of tasks that have been completed.
        """
        completed_tasks = []

        # Step the pending WorkProfiles and add the completed ones to the set of
        # available profiles.
        invalid_profiles = []
        for profile, loading_strategy in self._pending_profiles.items():
            remaining_time = loading_strategy.runtime - step_size
            if remaining_time <= EventTime.zero():
                # The WorkProfile has finished loading, make it available.
                self._available_profiles[profile] = ExecutionStrategy(
                    resources=loading_strategy.resources,
                    batch_size=loading_strategy.batch_size,
                    runtime=EventTime.zero(),
                )
                self._logger.debug(
                    "[%d] WorkProfile %s has finished loading on %s.",
                    current_time.to(EventTime.Unit.US).time,
                    profile.name,
                    self.name,
                )
                invalid_profiles.append(profile)
            else:
                # The WorkProfile has still not finished loading, update the time
                # remaining until it becomes available.
                loading_strategy._runtime = remaining_time

        # Remove the completed WorkProfiles from the set of pending profiles.
        for profile in invalid_profiles:
            del self._pending_profiles[profile]

        # Invoke the step() method on all the tasks.
        for task in self._placed_tasks:
            if task.state != TaskState.RUNNING:
                self._logger.debug(
                    f"[{current_time.to(EventTime.Unit.US).time}] Skipping stepping "
                    f"for Task {task.unique_name} because it is in state {task.state}."
                )
                continue
            if task.step(current_time, step_size):
                self._logger.debug(
                    f"[{current_time.to(EventTime.Unit.US).time}] {task.unique_name} "
                    f"finished execution on {self}."
                )
                completed_tasks.append(task)
        return completed_tasks

    def is_available(self, profile: WorkProfile) -> EventTime:
        """Check if the given `WorkProfile` is available on this `Worker`.

        The method returns an EventTime instance that denotes how far into the future
        a given profile will be available. An instance of `EventTime.invalid()` means
        that the `WorkProfile` has not been requested for loading onto this `Worker`,
        and an instance of `EventTime.zero()` means that the profile is available at
        the current time.

        Args:
            profile (`WorkProfile`): The profile whose availability needs to be checked.

        Returns:
            An `EventTime` instance specifying when the profile will be available at
            this `Worker`.
        """
        if profile in self._available_profiles:
            return EventTime.zero()
        elif profile in self._pending_profiles:
            return self._pending_profiles[profile].runtime
        else:
            return EventTime.invalid()

    def get_available_profiles(self) -> Sequence[WorkProfile]:
        """Get the set of `WorkProfile`s that are available on this `Worker` at the
        current time.

        Returns:
            A sequence of `WorkProfile`s that are available on this `Worker`.
        """
        return list(self._available_profiles.keys())

    def get_pending_profiles(self) -> Sequence[WorkProfile]:
        """Get the set of `WorkProfile`s that are pending to be loaded on this `Worker`
        at the current time.

        Returns:
            A sequence of `WorkProfile`s that are pending to be loaded on this `Worker`.
        """
        return list(self._pending_profiles.keys())

    def is_full(self) -> bool:
        """Check if the Worker is full.

        Returns:
            `True` if all the Resources in the Worker are full, `False` otherwise.
        """
        return self._resources.empty()

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

        # Copy the placed and pending profiles.
        for profile_id, profile in self._available_profiles.items():
            instance._available_profiles[profile_id] = profile
        for profile_id, profile in self._pending_profiles.items():
            instance._pending_profiles[profile_id] = profile

        # Copy the placed tasks.
        for task, strategy in self._placed_tasks.items():
            instance._placed_tasks[task] = strategy

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
    def resources(self) -> Resources:
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
            self._logger = setup_logging(name=self.__class__.__name__)

        self._name = name
        # A Mapping from the ID of the Worker to the instance of the Worker.
        self._workers: Mapping[str, Worker] = {worker.id: worker for worker in workers}
        self._scheduler = scheduler
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        # A mapping from the Task to the ID of the Worker.
        self._placed_tasks: Mapping[Task, str] = {}

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

    def place_task(
        self,
        task: Task,
        execution_strategy: Optional[ExecutionStrategy] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Places the task on this `WorkerPool`.

        The caller must ensure that the `WorkerPool` has enough resources to
        execute this task before invoking this method. Further, the
        `WorkerPool` is in charge of executing the task across its workers.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.
            execution_strategy (`Optional[ExecutionStrategy`]): The ExecutionStrategy
                to be used for placing this task. If no strategy is provided, any
                fit strategy is used.
            worker_id (`Optional[str]`): The ID of the Worker where the Task
                is to be placed on this WorkerPool. If `None` and a secondary
                scheduler is provided, the results from that scheduler are
                used. Otherwise, the task is placed on the first worker that
                can accomodate it.

        Returns:
            False if the task could not be placed due to insufficient resources.
        """
        placement, strategy = None, None
        if worker_id is not None:
            # If a WorkerID is provided, ensure that the Worker with that ID
            # is available on this WorkerPool.
            if worker_id not in self._workers:
                raise ValueError(
                    f"The WorkerID {worker_id} was not found in "
                    f"the WorkerPool {self.id}."
                )
            if execution_strategy and not self._workers[
                worker_id
            ].can_accomodate_strategy(execution_strategy):
                worker = self._workers[worker_id]
                error_message = (
                    f"{worker.name} ({worker_id}) cannot accomodate the strategy "
                    f"{execution_strategy} ({execution_strategy.id}) for task "
                    f"{task.unique_name}. It is running: "
                    f"[{','.join(t.unique_name for t in worker.get_placed_tasks())}]"
                )
                self._logger.error(error_message)
                return False
            else:
                strategy = execution_strategy

            if not execution_strategy:
                for possible_strategy in task.available_execution_strategies:
                    if self._workers[worker_id].can_accomodate_strategy(
                        possible_strategy
                    ):
                        strategy = possible_strategy
                        break
            placement = worker_id
        elif self._scheduler is not None:
            # If a scheduler was provided, get a task placement from it.
            placements = self._scheduler.schedule(
                sim_time=None,
                workload=Workload.from_task_graphs(
                    task_graphs={"test_task_graph": TaskGraph(tasks={task: []})}
                ),
                worker_pools=WorkerPools([self]),
            )
            # Add the runtime to the task start time.
            task._start_time += placements.runtime
            placement = placements.get_placement(task).worker_id
            strategy = placements.get_placement(task).execution_strategy
        elif execution_strategy is not None:
            # If there was no scheduler, find the first worker that can
            # accomodate the task given its resource requirements.
            for _id, _worker in self._workers.items():
                if _worker.can_accomodate_strategy(execution_strategy):
                    placement = _id
                    break
            strategy = execution_strategy
        else:
            # If there was no provided strategy, search if any strategy is executable
            # on any of the Workers.
            for _id, _worker in self._workers.items():
                for possible_strategy in task.available_execution_strategies:
                    if _worker.can_accomodate_strategy(possible_strategy):
                        placement = _id
                        strategy = possible_strategy
                        break
                if placement is not None:
                    break

        if placement is None:
            self._logger.warning(
                f"The {task} could not be placed on {self.id} WorkerPool."
            )
            return False
        else:
            self._workers[placement].place_task(task, strategy)
            self._placed_tasks[task] = placement
            return True

    def remove_task(self, current_time: EventTime, task: Task):
        """Removes the task from this `WorkerPool`.

        Args:
            current_time (`EventTime`): The time at which the removal was requested.
            task (`Task`): The task to be placed on this `WorkerPool`.

        Raises:
            `ValueError` if the task was not placed on this worker pool.
        """
        if task not in self._placed_tasks:
            raise ValueError(f"The task {task} was not placed on {self.id} WorkerPool.")

        # Deallocate the resources and remove the placed task.
        self._workers[self._placed_tasks[task]].remove_task(
            current_time=current_time, task=task
        )
        del self._placed_tasks[task]

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s that are currently placed on this `WorkerPool`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        return list(self._placed_tasks.keys())

    def step(
        self,
        current_time: EventTime,
        step_size: EventTime = EventTime(1, EventTime.Unit.US),
    ) -> Sequence[Task]:
        """Steps all the tasks of this `WorkerPool` by the given `step_size`.

        Args:
            current_time (`int`): The current time of the simulator  (in us).
            step_size (`int`): The amount of time for which to step the
                workers (in us).

        Returns:
            The set of tasks that have finished execution.
        """
        completed_tasks = []
        for _, worker in self._workers.items():
            self._logger.debug(
                f"Stepping through the execution of {worker} for {step_size} "
                f"steps from time {current_time}"
            )
            completed_tasks.extend(worker.step(current_time, step_size))
        return completed_tasks

    def can_accomodate_strategy(self, execution_strategy: ExecutionStrategy) -> bool:
        """Checks if any of the `Worker`s of this `WorkerPool` can accomodate
        the given `ExecutionStrategy` based on its resource availability.

        Args:
            execution_strategy (`ExecutionStrategy`): The execution strategy to be used
                to place a Task on this `WorkerPool`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        return any(
            worker.can_accomodate_strategy(execution_strategy)
            for worker in self._workers.values()
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

    def is_full(self) -> bool:
        """Check if the WorkerPool is full.

        Returns:
            `True` if all the Resources in the WorkerPool are full, `False` otherwise.
        """
        return all(worker.is_full() for worker in self._workers.values())

    def evict_profile(
        self, profile: WorkProfile, worker_id: Optional[str] = None
    ) -> None:
        """Evicts the `profile` from the requested `Worker`s on this `WorkerPool`
        instance. If no `Worker`s are specified, the `profile` is evicted from all
        the `Worker`s on this `WorkerPool`.

        Args:
            profile (`WorkProfile`): The `Profile` to be evicted.
            worker_id (`Optional[str]`): The ID of the `Worker` from which the profile
                is to be evicted.
        """
        worker_ids = (
            [worker_id] if worker_id else [worker.id for worker in self.workers]
        )
        for worker in worker_ids:
            self._workers[worker].evict_profile(profile)

    def load_profile(
        self,
        profile: WorkProfile,
        loading_strategy: ExecutionStrategy,
        worker_id: Optional[str] = None,
    ) -> None:
        """Loads the given `profile` into the set of available profiles at the
        requested `Worker`. If no `Worker` is specified, the `profile` is loaded into
        all the `Worker`s on this `WorkerPool`.

        Args:
            profile (`WorkProfile`): The `Profile` to be loaded.
            loading_strategy (`ExecutionStrategy`): The `ExecutionStrategy` to be used
                to load the profile onto the `Worker`.
            worker_id (`Optional[str]`): The ID of the `Worker` into which the profile
                is to be loaded.
        """
        worker_ids = (
            [worker_id] if worker_id else [worker.id for worker in self.workers]
        )
        for worker in worker_ids:
            self._workers[worker].load_profile(profile, loading_strategy)

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
        if self._logger.isEnabledFor(logging.DEBUG):
            return f"WorkerPool(name={self.name}, id={self.id})"
        else:
            return self.name

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
    """A collection of `WorkerPool`s.

    Args:
        worker_pools (`Sequence[WorkerPool]`): A sequence of `WorkerPool`s
            that are contained in this instance.
    """

    def __init__(self, worker_pools: Sequence[WorkerPool]) -> None:
        self._worker_pools = {
            worker_pool.id: worker_pool for worker_pool in worker_pools
        }

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s placed across all the `Worker`s.

        Returns:
            A `Sequence[Task]` representing the tasks placed across the
            `Worker`s in this instance.
        """
        placed_tasks = []
        for wp in self.worker_pools:
            placed_tasks.extend(wp.get_placed_tasks())
        return placed_tasks

    def get_worker_pool(self, worker_pool_id: str) -> Optional[WorkerPool]:
        """Retrieve the WorkerPool with the given ID from this collection.

        Args:
            worker_pool_id (`str`): The ID of the WorkerPool to retrieve.

        Returns:
            The `WorkerPool` instance with the given ID if found, `None` otherwise.
        """
        return self._worker_pools.get(worker_pool_id)

    def is_full(self) -> bool:
        """Check if the WorkerPools is full.

        Returns:
            `True` if all the WorkerPools are full, `False` otherwise.
        """
        return all(worker_pool.is_full() for worker_pool in self._worker_pools.values())

    @property
    def worker_pools(self) -> Sequence[WorkerPool]:
        """Retrieve the collection of :py:class:`WorkerPool` instances stored in this
        instance.

        Returns:
            A `Sequence[WorkerPool]` stored in this instance of WorkerPools.
        """
        return self._worker_pools.values()

    def __copy__(self):
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, [copy(wp) for wp in self.worker_pools])
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, [deepcopy(wp) for wp in self.worker_pools])
        return instance

    def __len__(self):
        return len(self._worker_pools)
