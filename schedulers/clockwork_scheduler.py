import time
from copy import copy
from typing import Mapping, Optional, Sequence, Set

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import ExecutionStrategy, Placement, Placements, Task, TaskState, Workload
from workload.profile import WorkProfile


class Models:
    """Defines a priority order over the models to figure out which
    needs to be loaded and evicted."""


class Model:
    def __init__(self, profile: WorkProfile):
        self._task_priority_queue: Sequence[Task] = []
        self._profile = profile

    @property
    def id(self) -> str:
        return self._profile.id

    @property
    def profile(self) -> WorkProfile:
        return self._profile

    def add_task(self, task: Task):
        if task not in self._task_priority_queue:
            self._task_priority_queue = self._task_priority_queue + [task]

    def get_feasible_placements(
        self,
        sim_time: EventTime,
        worker_pool_id: str,
        worker: Worker,
    ) -> Sequence[Placement]:
        """Returns a feasible execution strategy with the largest batch size
        that meets the earliest deadline, and removes all tasks from the queues
        that comprise the batch."""
        if len(self._task_priority_queue) == 0 or not worker.is_available(self.profile):
            return []

        # TODO: Update this when the task PQ implementation changes.
        deadline = self._task_priority_queue[0].deadline

        strategies = worker.get_compatible_strategies(self.profile.execution_strategies)
        chosen_strategy = strategies.get_fastest_strategy()
        if chosen_strategy is None:
            return []
        for strategy in strategies:
            if sim_time + strategy.runtime <= deadline:
                if (
                    strategy.batch_size > chosen_strategy.batch_size
                    # Prevent over-provisioning for an unnecessarily large batch size.
                    and chosen_strategy.batch_size < len(self._task_priority_queue)
                ):
                    chosen_strategy = strategy

        placements = []
        for _ in range(chosen_strategy.batch_size):
            # TODO: Update this when the task PQ implementation changes.
            task = self._task_priority_queue[0]
            self._task_priority_queue = self._task_priority_queue[1:]

            placements.append(
                Placement.create_task_placement(
                    task, sim_time, worker_pool_id, worker.id, chosen_strategy
                )
            )
        return placements


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
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(ClockworkScheduler, self).__init__(
            runtime=runtime,
            enforce_deadlines=True,
            _flags=_flags,
        )
        self._log_to_file = log_to_file

        self._models: Mapping[str, Model] = {}
        # Maps worker ID to model ID.
        # LRU cache of models loaded onto each GPU.
        self._cached_models: Mapping[int, Sequence[str]] = {}
        # Models not loaded onto each GPU, ordered by priority.
        self._not_cached_models: Mapping[int, Sequence[str]] = {}

    def start(
        self,
        start_time: EventTime,
        work_profiles: Set[WorkProfile],
        worker_pools: "WorkerPools",
    ) -> Placements:
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

        # Construct a mapping of the index of a Worker to the Worker itself, and
        # a mapping from the Worker to the WorkerPool to which it belongs.
        worker_index = 1
        workers: Mapping[str, Worker] = {}
        worker_to_worker_pool = {}
        for worker_pool in schedulable_worker_pools.worker_pools:
            for worker in worker_pool.workers:
                workers[worker_index] = worker
                worker_index += 1
                worker_to_worker_pool[worker.id] = worker_pool.id

        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks for scheduling across {len(workers)} workers. These tasks were: "
            f"{[task.unique_name for task in tasks_to_be_scheduled]}."
        )

        # Populate the models with tasks.
        scheduler_start_time = time.time()
        placements = []
        for task in tasks_to_be_scheduled:
            model_id = task.profile.id
            if model_id not in self._models:
                self._models[model_id] = Model(model_id)
            # Admission control
            if task.deadline > sim_time and self.enforce_deadlines:
                placements.append(
                    Placement(
                        task=task,
                        placement_time=None,
                        worker_pool_id=None,
                        worker_id=None,
                        execution_strategy=None,
                    )
                )
            else:
                if model_id in self._models:
                    self._models[model_id].add_task(task)
                else:
                    model = Model(task.profile)
                    model.add_task(task)
                    self._models[model_id] = model

        for worker in workers.values():
            # Skip workers currently processing tasks.
            if len(worker.get_placed_tasks()) > 0:
                continue
            for model in self._models.values():
                new_placements = model.get_feasible_placements(
                    sim_time, worker_to_worker_pool[worker.id], worker.id
                )
                placements.extend(new_placements)

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
