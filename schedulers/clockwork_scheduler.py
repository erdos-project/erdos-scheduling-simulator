import bisect
import time
from collections import deque
from copy import copy
from operator import attrgetter
from typing import Callable, Iterator, List, Optional, Sequence, Set, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import (
    BatchStrategy,
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    Workload,
)
from workload.profile import WorkProfile


class Model(object):
    """A representation of a model that is loaded onto a Worker.

    This instance maintains a request queue that sorts the remaining tasks by their
    deadline. The `get_feasible_placements` method returns a feasible execution strategy
    with the highest batch size that meets the earliest deadline, and removes all tasks
    that are past its deadline but couldn't be scheduled.

    Args:
        profile (`WorkProfile`): The WorkProfile of the model that provides the loading
            and execution strategies.
    """

    def __init__(self, profile: WorkProfile):
        self._tasks: Set[Task] = set()
        self._task_priority_queue: List[Task] = []
        self._profile = profile

    @property
    def id(self) -> str:
        return self._profile.id

    @property
    def profile(self) -> WorkProfile:
        return self._profile

    @property
    def outstanding_requests(self) -> int:
        """Returns the number of requests available in the queue for this `Model`."""
        return len(self._tasks)

    def get_available_execution_strategies(
        self, current_time: EventTime, key: Callable = attrgetter("batch_size")
    ) -> ExecutionStrategies:
        """Retrieve the strategies that are available for execution given the current
        set of requests available in the model. The strategies are ordered by the
        provided function. The default ordering is by the batch size.

        Args:
            current_time (`EventTime`): The current time of the simulation.
            key (`Callable`): The function to use for ordering the strategies.

        Returns:
            An `ExecutionStrategies` object containing the available strategies sorted
            by their priority as defined by the `key` function.
        """
        available_strategies = ExecutionStrategies()
        for strategy in sorted(
            self._profile.execution_strategies, key=key, reverse=True
        ):
            # The criteria for assuming the strategy is valid are as follows:
            # 1. The number of requests in the queue must saturate the batch size of
            #    the strategy.
            # 2. The execution time of the strategy must be able to meet the deadline
            #    of the earliest request in the queue.
            if strategy.batch_size <= self.outstanding_requests and (
                current_time + strategy.runtime <= self._task_priority_queue[0].deadline
            ):
                available_strategies.add_strategy(strategy)
        return available_strategies

    def add_task(self, task: Task) -> None:
        """Adds a new Task to the request queue, while keeping the request queue sorted
        by the deadline.

        If the `Task` is already available in the queue, it is not added again.

        Args:
            task (`Task`): The task to add for scheduling.

        Raises:
            A `ValueError` if the `Task`'s profile does not match the model's profile.
        """
        if task.profile != self._profile:
            raise ValueError(
                f"The task's profile ({task.profile.id}) does not match the model's "
                f"profile ({self._profile.id})."
            )
        if task not in self._tasks:
            self._tasks.add(task)
            bisect.insort(self._task_priority_queue, task)

    def remove_task(self, task: Task) -> None:
        """Removes a task from the request queue.

        Args:
            task (`Task`): The task to remove from the request queue.
        """
        if task in self._tasks:
            self._tasks.remove(task)
            self._task_priority_queue.remove(task)

    def get_placements(
        self,
        sim_time: EventTime,
        strategy: ExecutionStrategy,
        worker_pool_id: str,
        worker_id: str = None,
    ) -> Sequence[Placement]:
        """Returns a sequence of `Placement`s that contain the `Task`s that can be
        batched together to meet the requirements of the provided `ExecutionStrategy`.

        The method internally constructs a `BatchStrategy` object corresponding to the
        provided `ExecutionStrategy` and uses it to inform the `Worker` to batch the
        tasks in the request queue. The `Task`s are then removed from the request queue.

        Args:
            sim_time (`EventTime`): The current time of the simulation. This is used to
                inform the placement time of the `Placement` objects.
            strategy (`ExecutionStrategy`): The strategy to use for batching the tasks.
            worker_pool_id (`str`): The ID of the `WorkerPool` that the `Task`s need
                to be placed on.
            worker_id (`str`): The ID of the `Worker` that the `Task`s need to be placed
                on. If `None`, the `WorkerPool` finds appropriate `Worker`s to place the
                `Task`s on.

        Returns:
            A sequence of `Placement` objects to be returned by the scheduler.

        Raises:
            A `RuntimeError` if the number of tasks in the queue is less than the batch
            size of the provided `ExecutionStrategy`.
        """
        if len(self._task_priority_queue) < strategy.batch_size:
            raise RuntimeError(
                f"The number of tasks in the queue ({len(self._task_priority_queue)}) "
                f"is less than the batch size ({strategy.batch_size})."
            )

        # Construct the `Placement` objects.
        batch_strategy = BatchStrategy(execution_strategy=strategy)
        tasks = self._task_priority_queue[: strategy.batch_size]
        placements = []
        for task in tasks:
            placement = Placement.create_task_placement(
                task=task,
                placement_time=sim_time,
                worker_pool_id=worker_pool_id,
                worker_id=worker_id,
                execution_strategy=batch_strategy,
            )
            placements.append(placement)
            self._tasks.remove(task)

        # Update the priority queue.
        self._task_priority_queue = self._task_priority_queue[strategy.batch_size :]

        return placements


class Models:
    """Encapsulates individual instances of `Model`s available during the simulation,
    and maintains a priority queue over the models to figure out which models need to
    be loaded and evicted.

    Args:
        models (`Sequence[Model]`): The models available during the simulation.
    """

    def __init__(self, models: Sequence[Model] = []) -> None:
        self._models = {model.id: model for model in models}

    def add_model(self, model: Model) -> None:
        """Adds a new model to the set of models available during the simulation.

        Args:
            model (`Model`): The model to add.
        """
        if model.id not in self._models:
            self._models[model.id] = model

    def add_task(self, task: Task) -> None:
        """Adds a new task to the request queue of the corresponding model.

        If the model corresponding to the `Task`s profile has not been seen before, a
        new instance of `Model` is created and added to the set of available models.
        If the `Task` is already available in the queue, it is not added again.

        Args:
            task (`Task`): The task to add.
        """
        if task.profile.id not in self._models:
            self.add_model(Model(task.profile))
        self._models[task.profile.id].add_task(task)

    def __iter__(self) -> Iterator[Model]:
        for model in self._models.values():
            yield model

    def __getitem__(self, model_id: str) -> Model:
        return self._models[model_id]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models


class ClockworkScheduler(BaseScheduler):
    """Implements the Clockwork infer5 scheduling algorithm.

    Args:
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        log_to_file (`bool`): If `True`, the scheduler writes the Gurobi search
            log to files with the format "gurobi_{sim_time}.log".
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
    """

    def __init__(
        self,
        runtime: EventTime = EventTime.invalid(),
        _flags: Optional["absl.flags"] = None,
    ):
        super(ClockworkScheduler, self).__init__(
            runtime=runtime,
            enforce_deadlines=True,
            _flags=_flags,
        )

        # Maintain a set of `Model` instances to keep track of the request queues and
        # construct batching strategies.
        self._models = Models()

    def start(
        self,
        start_time: EventTime,
        work_profiles: Set[WorkProfile],
        worker_pools: "WorkerPools",
    ) -> Placements:
        # Maintain an instance of `Model` for each provided `WorkProfile` to keep track
        # of the request queues and construct batching strategies.
        for work_profile in work_profiles:
            self._models.add_model(Model(work_profile))

        # Clockwork relies on the Client to initiate `LoadRemoteModel` requests to the
        # controller during the startup phase. The `ControllerStartup` then enqueus
        # the loading of each of the models requested onto all of the workers in the
        # system. We simulate this by trying to load all of the models specified in the
        # Workload on top of all the Workers, and log an error if the resource
        # requirements cannot be met.
        start_time = time.time()
        placements = []
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                # We create a virtual copy of the Worker to simulate the loading of
                # each model onto the Worker to correctly account for the resources.
                worker_copy = copy(worker)
                for work_profile in work_profiles:
                    if worker_copy.is_available(work_profile) == EventTime.invalid():
                        compatible_strategies = worker_copy.get_compatible_strategies(
                            work_profile.loading_strategies
                        )
                        if len(compatible_strategies) == 0:
                            self._logger.warning(
                                "[%s] No compatible strategies found for loading "
                                "WorkProfile %s onto Worker %s",
                                start_time,
                                work_profile.id,
                                worker.id,
                            )
                            continue
                        # We do not expect more than one strategy to be compatible
                        # with each Worker.
                        initial_loading_strategy = (
                            compatible_strategies.get_fastest_strategy()
                        )

                        # We load the model onto the Worker with an amended strategy
                        # such that all the models are available by the time the first
                        # event executes in the Simulator. This replicates the
                        # ControllerWithStartupPhase behavior in Clockwork.
                        adjusted_loading_strategy = ExecutionStrategy(
                            resources=initial_loading_strategy.resources,
                            batch_size=initial_loading_strategy.batch_size,
                            runtime=EventTime.zero(),
                        )
                        placements.append(
                            Placement.create_load_profile_placement(
                                work_profile=work_profile,
                                placement_time=start_time,
                                worker_pool_id=worker_pool.id,
                                worker_id=worker.id,
                                loading_strategy=adjusted_loading_strategy,
                            )
                        )
                        worker_copy.load_profile(
                            profile=work_profile,
                            loading_strategy=adjusted_loading_strategy,
                        )
                        self._logger.debug(
                            "[%s] Requesting to load WorkProfile %s onto Worker %s.",
                            start_time,
                            work_profile.id,
                            worker.id,
                        )
        end_time = time.time()
        return Placements(
            runtime=EventTime.zero(),
            true_runtime=EventTime(
                int((end_time - start_time) * 1e6), EventTime.Unit.US
            ),
            placements=placements,
        )

    def process_model_strategies(
        self,
        current_time: EventTime,
        model: Model,
        execution_strategies: ExecutionStrategies,
        worker_pools: WorkerPools,
    ) -> Sequence[Placement]:
        """Processes the execution strategies for the given model.

        If the method returns a non-empty `Sequence` of `Placement` instances, the
        method was able to find a possible instantiation of one of the strategies on
        some `Worker`. The method is responsible for making sure that the resources
        for the `Worker` are correctly accounted for. However, the caller must check
        the return value to see if more tasks from the same `Model` can be scheduled.

        Args:
            current_time (`EventTime`): The current time of the simulation.
            model (`Model`): The model to process the execution strategies for.
            execution_strategies (`ExecutionStrategies`): The execution strategies
                to process.
            worker_pools (`WorkerPools`): The worker pools to use for determining the
                placement of the execution strategies.

        Returns:
            A (possibly empty) `Sequence` of `Placement` instances that represent the
            placement of the set of `Task`s that correspond to that execution strategy.
        """
        placements = []
        for execution_strategy in execution_strategies:
            # We try to find a WorkerPool that has a Worker that can execute the
            # strategy.
            for worker_pool in worker_pools.worker_pools:
                for worker in worker_pool.workers:
                    if worker.is_available(model.profile) != EventTime.zero():
                        self._logger.debug(
                            "[%s] Skipping Worker %s since the Model %s (%s) is not "
                            "loaded. The model will be available in %s.",
                            current_time.to(EventTime.Unit.US).time,
                            worker,
                            model.profile.name,
                            model.profile.id,
                            worker.is_available(model.profile),
                        )
                        break
                    if worker.can_accomodate_strategy(execution_strategy):
                        task_placements: Sequence[Placement] = model.get_placements(
                            sim_time=current_time,
                            strategy=execution_strategy,
                            worker_pool_id=worker_pool.id,
                            worker_id=worker.id,
                        )
                        for placement in task_placements:
                            placements.append(placement)
                            worker.place_task(
                                task=placement.task,
                                execution_strategy=placement.execution_strategy,
                            )
                            self._logger.debug(
                                "[%s] Requesting to execute Task %s with Model %s on "
                                "Worker %s with the strategy %s.",
                                current_time.to(EventTime.Unit.US).time,
                                placement.task,
                                model.profile,
                                worker,
                                placement.execution_strategy,
                            )
                        return placements
                    else:
                        self._logger.debug(
                            "[%s] Skipping Worker %s since it cannot meet the resource "
                            "requirements for the strategy %s. The strategy requires "
                            "%s, but the Worker has %s.",
                            current_time.to(EventTime.Unit.US).time,
                            worker,
                            execution_strategy,
                            execution_strategy.resources,
                            worker.resources,
                        )
        return placements

    def run_admission(
        self, current_time: EventTime, tasks_to_schedule: Sequence[Task]
    ) -> Sequence[Placement]:
        """Runs admission control on the given set of tasks. This method adds the
        `Task`s to the request queue of the corresponding models and returns a
        failed `Placement` for each `Task` that is past its deadline.

        This method roughly corresponds to the `run_admission_thread` method of the
        infer5 scheduler in Clockwork [here](https://tinyurl.com/45evsxbp).

        Args:
            current_time (`EventTime`): The current time of the simulation.
            tasks_to_schedule (`Sequence[Task]`): The set of `Task`s available for
                execution.

        Returns:
            A (possibly empty) `Sequence` of `Placement` instances that represent the
            failed `Task` placements.
        """
        placements = []
        for task in tasks_to_schedule:
            if task.deadline < current_time and self.enforce_deadlines:
                placements.append(Placement.create_task_placement(task=task))
                self._logger.debug(
                    "[%s] Task %s has a deadline of %s, which has been missed. ",
                    current_time.to(EventTime.Unit.US).time,
                    task,
                    task.deadline.to(EventTime.Unit.US).time,
                )
            else:
                self._models.add_task(task)
                self._logger.debug(
                    "[%s] Added Task %s to the request queue for Model %s.",
                    current_time.to(EventTime.Unit.US).time,
                    task,
                    task.profile.name,
                )
        return placements

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
        )

        # Create a virtual WorkerPool set to try scheduling decisions on.
        schedulable_worker_pools = copy(worker_pools)
        num_workers = sum(
            len(worker_pool.workers)
            for worker_pool in schedulable_worker_pools.worker_pools
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks for scheduling across {num_workers} workers. "
            f"These tasks were: {[task.unique_name for task in tasks_to_be_scheduled]}."
        )

        scheduler_start_time = time.time()
        placements = []

        # Run admission control on the `Task`s available for execution.
        failed_admission_control_tasks = self.run_admission(
            current_time=sim_time, tasks_to_schedule=tasks_to_be_scheduled
        )
        if len(failed_admission_control_tasks) > 0:
            self._logger.debug(
                "[%s] A total of %d tasks (out of %d) failed admission control.",
                sim_time.to(EventTime.Unit.US).time,
                len(failed_admission_control_tasks),
                len(tasks_to_be_scheduled),
            )
            placements.extend(failed_admission_control_tasks)

        # Retrieve the available strategies for each `Model` instance, ordered by their
        # execution preference, and find a `Worker` that can execute the strategy.
        execution_strategies_queue: Sequence[
            Tuple[Model, ExecutionStrategies]
        ] = deque()
        for model in self._models:
            model_strategies = model.get_available_execution_strategies(
                current_time=sim_time
            )
            if len(model_strategies) > 0:
                execution_strategies_queue.append((model, model_strategies))

        while len(execution_strategies_queue) > 0:
            model, model_strategies = execution_strategies_queue.popleft()
            model_placements = self.process_model_strategies(
                current_time=sim_time,
                model=model,
                execution_strategies=model_strategies,
                worker_pools=schedulable_worker_pools,
            )
            if len(model_placements) > 0:
                # There were some placements for the model. We add the model back to
                # the queue, so that we can try to schedule more tasks for it.
                placements.extend(model_placements)
                new_model_strategies = model.get_available_execution_strategies(
                    current_time=sim_time
                )
                if len(new_model_strategies) > 0:
                    execution_strategies_queue.append((model, new_model_strategies))
            else:
                self._logger.debug(
                    "[%s] No valid placements were found for Model %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    model.profile.name,
                )

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        self._logger.debug(
            f"[{sim_time.time}] The runtime of the scheduler was: {scheduler_runtime}."
        )
        runtime = (
            scheduler_runtime
            if self.runtime == EventTime(-1, EventTime.Unit.US)
            else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )
