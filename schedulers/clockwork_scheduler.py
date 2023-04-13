from copy import copy
import time
from typing import Mapping, Optional, Sequence, Tuple, Union

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import Placement, Placements, Task, TaskState, Workload
from workload.profile import WorkProfile
from workload.strategy import ExecutionStrategy


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
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None. Likewise, tasks which cannot meet their
            deadlines will not be placed.
        log_to_file (`bool`): If `True`, the scheduler writes the Gurobi search
            log to files with the format "gurobi_{sim_time}.log".
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
    """

    def __init__(
        self,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        enforce_deadlines: bool = False,
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(ClockworkScheduler, self).__init__(
            runtime=runtime,
            enforce_deadlines=enforce_deadlines,
            _flags=_flags,
        )
        self._log_to_file = log_to_file

        self._models: Mapping[str, Model] = {}
        # Maps worker ID to model ID.
        # LRU cache of models loaded onto each GPU.
        self._cached_models: Mapping[int, Sequence[str]] = {}
        # Models not loaded onto each GPU, ordered by priority.
        self._not_cached_models: Mapping[int, Sequence[str]] = {}

    def start(self, start_time: EventTime) -> None:
        raise NotImplementedError

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

        # If we are not retracting schedules, we should consider both
        # RUNNING and SCHEDULED task placements as permanent.
        filter_fn = lambda task: task.state in (  # noqa: E731
            TaskState.RUNNING,
            TaskState.SCHEDULED,
        )
        previously_placed_tasks = workload.filter(filter_fn)

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
        self._logger.debug(
            f"[{sim_time.time}] The scheduler is also considering the following "
            f"{len(previously_placed_tasks)} for their effects on the current "
            f"placements: {[task.unique_name for task in previously_placed_tasks]}."
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
