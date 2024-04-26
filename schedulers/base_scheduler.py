import time
from copy import copy
from operator import itemgetter
from typing import Optional, Sequence, Set

import absl  # noqa: F401

from utils import EventTime, setup_logging
from workers import WorkerPool, WorkerPools
from workload import (
    BranchPredictionPolicy,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
    WorkProfile,
)


class BaseScheduler(object):
    """Base class that the schedulers must implement.

    Args:
        preemptive (`bool`): If `True`, the scheduler can preempt the tasks that
            are currently running.
        runtime (`EventTime`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
        lookahead (`EventTime`): The scheduler will try to place tasks that are within
            the scheduling lookahead (in us) using estimated task release times.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        policy (`BranchPredictionPolicy`): The branch prediction policy to use for the
            scheduler if it schedules future tasks.
        branch_prediction_accuracy (`float`): The accuracy of the branch prediction
            policy used by the scheduler.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
        release_taskgraphs (`bool`): If `True`, the scheduler is given access to the
            entire TaskGraph for any Task that falls within the lookahead defined above.
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        policy: BranchPredictionPolicy = BranchPredictionPolicy.RANDOM,
        branch_prediction_accuracy: float = 0.50,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        self._preemptive = preemptive
        self._runtime = runtime
        self._lookahead = lookahead
        self._enforce_deadlines = enforce_deadlines
        self._policy = policy
        self._branch_prediction_accuracy = branch_prediction_accuracy
        self._retract_schedules = retract_schedules
        self._release_taskgraphs = release_taskgraphs
        self._flags = _flags

        if self._flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=self._flags.log_file_name,
                log_level=self._flags.log_level,
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
        
        print(f"self._logger: {self._logger}, {self._logger.handlers}")
        self._logger.debug("TEST print")

    def start(
        self,
        start_time: EventTime,
        work_profiles: Set[WorkProfile],
        worker_pools: "WorkerPools",  # noqa: F821
    ) -> Placements:
        """Initializes the startup phase of a scheduler. This method is invoked before
        requests start arriving to the Scheduler, and allows it to do basic set-up work.

        By default, the method tries to place all the work profiles on all the Workers
        in the system, and does not place a profile that cannot be accomodated on the
        Worker.

        Args:
            start_time (`EventTime`): The time at which the scheduler was invoked.
            work_profiles (`Set[WorkProfile]`): A (possibly) empty set of
                `WorkProfile`s provided to the scheduler at the start.
            worker_pools (`WorkerPools`): The set of worker pools available to the
                Scheduler at the initialization of the simulation.

        Returns:
            A `Placements` object that signifies the initial placements to be made by
            the Scheduler.
        """
        self._logger.info(
            "[%s] Initiated the %s with the default startup implementation "
            "with the following work profiles: [%s] on the worker pools [%s].",
            start_time.to(EventTime.Unit.US).time,
            self.__class__.__name__,
            ", ".join(work_profile.name for work_profile in work_profiles),
            ", ".join(worker_pool.name for worker_pool in worker_pools.worker_pools),
        )

        # Create a copy of the WorkerPools to try placement decisions on.
        worker_pools_copy = copy(worker_pools)

        scheduler_start_time = time.time()
        placements = []
        for worker_pool in worker_pools_copy.worker_pools:
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
                            "[%s] Requesting to load WorkProfile %s "
                            "(%s) onto Worker %s.",
                            start_time.to(EventTime.Unit.US).time,
                            work_profile.name,
                            work_profile.id,
                            worker.id,
                        )
        scheduler_end_time = time.time()
        return Placements(
            runtime=EventTime.zero(),
            true_runtime=EventTime(
                int((scheduler_end_time - scheduler_start_time) * 1e6),
                EventTime.Unit.US,
            ),
            placements=placements,
        )

    def notify_workload_updated(
        self,
        sim_time: EventTime,
        workload: Workload,
        worker_pools: "WorkerPools",  # noqa: F821
    ) -> None:
        """Notifies the Scheduler that the Workload has been updated.

        The Simulator invokes this method to notify the Scheduler that it can do
        any necessary bookkeeping or updates to its internal state.

        Args:
            sim_time (`EventTime`): The time at which the workload was updated.
            workload (`Workload`): The `Workload` that was updated.
            worker_pools (`WorkerPools`): The set of worker pools available to the
                Scheduler at the update of the Workload.
        """
        pass

    def schedule(
        self,
        sim_time: EventTime,
        workload: Workload,
        worker_pools: "WorkerPools",  # noqa: F821
    ) -> Placements:
        """Abstract method to be implemented by derived classes to allow the
        scheduling of tasks.

        Args:
            sim_time (`EventTime`): The time in us at which the scheduler is invoked.
            workload (`Workload`): The `Workload` at this invocation of the scheduler.
            worker_pools (`WorkerPools`): The set of worker pools to
                do the placement across.

        Returns:
            A representation of the Placements computed by the Scheduler.
        """
        raise NotImplementedError(
            "The `schedule()` method has not been " "implemented."
        )

    def log(self):
        raise NotImplementedError("The `log()` method has not been implemented.")

    @property
    def preemptive(self) -> bool:
        return self._preemptive

    @property
    def runtime(self) -> EventTime:
        return self._runtime

    @property
    def enforce_deadlines(self) -> bool:
        return self._enforce_deadlines

    @property
    def lookahead(self) -> EventTime:
        return self._lookahead

    @property
    def policy(self) -> BranchPredictionPolicy:
        return self._policy

    @property
    def retract_schedules(self) -> bool:
        return self._retract_schedules

    @property
    def release_taskgraphs(self) -> bool:
        return self._release_taskgraphs

    @property
    def branch_prediction_accuracy(self) -> float:
        return self._branch_prediction_accuracy

    def verify_schedule(
        self,
        sim_time: EventTime,
        workload: Workload,
        worker_pools: WorkerPools,
        placements: Placements,
    ) -> None:
        """Verifies a schedule returned by any scheduler.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            workload (`Workload`): The workload that was passed to the scheduler.
            worker_pools (`WorkerPools`): The worker pools that the tasks are supposed
                to be scheduled on.
            placements (`Placements`): The placements of the Tasks computed by the
                scheduler
        """
        for placement in placements:
            if not placement.is_placed():
                self._logger.debug(
                    "[%d] Any checks for task %s were skipped since it was not placed.",
                    sim_time.to(EventTime.Unit.US).time,
                    placement.task.unique_name,
                )
                continue

            # Ensure that the task starts after the time at which the scheduler
            # was invoked and after its release time.
            if (
                placement.placement_time < sim_time
                or placement.placement_time < placement.task.release_time
            ):
                self._logger.error(
                    "[%d] The task %s was placed at %s, but was released at %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    placement.task.unique_name,
                    placement.placement_time,
                    placement.task.release_time,
                )
                raise ValueError(
                    f"The placement for task {placement.task.unique_name} did not "
                    f"satisfy the lower bounds for placement time."
                )

            # Ensure that the task is expected to finish before its deadline if the
            # scheduler was required to enforce deadlines.
            if (
                self.enforce_deadlines
                and placement.placement_time + placement.task.remaining_time
                > placement.task.deadline
            ):
                self._logger.error(
                    "[%d] The task %s was placed at %s and has %s remaining time, "
                    "but has a deadline of %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    placement.task.unique_name,
                    placement.placement_time,
                    placement.task.remaining_time,
                    placement.task.deadline,
                )
                raise ValueError(
                    f"The placement for task {placement.task.unique_name} did not "
                    f"respect its deadline."
                )

            # Ensure that the task is only placed if all its parents are placed.
            task_graph = workload.get_task_graph(placement.task.task_graph)
            parents = task_graph.get_parents(placement.task)
            for parent in parents:
                any_parent_placed = False
                parent_placement = placements.get_placements(parent)[0]
                if parent.state in (
                    TaskState.VIRTUAL,
                    TaskState.RELEASED,
                    TaskState.PREEMPTED,
                ) or (
                    parent.state == TaskState.SCHEDULED and parent_placement is not None
                ):
                    # If the parent was supposed to be scheduled by this run, ensure
                    # that all the dependencies are being satisfied.
                    if (
                        parent_placement is None or not parent_placement.is_placed()
                    ) and not placement.task.terminal:
                        self._logger.error(
                            "[%d] The task %s was placed without a valid placement "
                            "for a parent task %s in state %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            parent.unique_name,
                            parent.state,
                        )
                        raise ValueError(
                            f"The placement for {placement.task.unique_name} did not "
                            f"respect the task dependencies."
                        )

                    if placement.task.terminal and (
                        parent_placement is None or not parent_placement.is_placed()
                    ):
                        continue

                    if (
                        parent_placement.placement_time + parent.remaining_time
                        >= placement.placement_time
                    ):
                        self._logger.error(
                            "[%d] The task %s was placed at %s before the parent %s "
                            "(placed at %s with remaining time of %s) finishes at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            placement.placement_time,
                            parent.unique_name,
                            parent_placement.placement_time,
                            parent.remaining_time,
                            parent_placement.placement_time + parent.remaining_time,
                        )
                        raise ValueError(
                            f"The placement for {placement.task.unique_name} did not "
                            f"respect the task dependencies."
                        )
                    else:
                        any_parent_placed = True
                elif parent.state == TaskState.SCHEDULED and parent_placement is None:
                    # If the task was scheduled previously, and was not reconsidered
                    # for scheduling, use the previously decided start time.
                    if (
                        parent.expected_start_time + parent.remaining_time
                        <= placement.placement_time
                    ):
                        self._logger.error(
                            "[%d] The task %s was placed at %s before a previously "
                            "scheduled parent %s (with an expected start time of %s "
                            "and remaining time of %s) finishes at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            parent.unique_name,
                            parent.expected_start_time,
                            parent.remaining_time,
                            parent.expected_start_time + parent.remaining_time,
                        )
                        raise ValueError(
                            f"The placement for {placement.task.unique_name} did not "
                            f"respect the task dependencies."
                        )
                    else:
                        any_parent_placed = True
                elif parent.state == TaskState.RUNNING:
                    # If the task is currently running, use the remaining time to
                    # ensure that the dependencies were respected.
                    if sim_time + parent.remaining_time <= placement.placement_time:
                        self._logger.error(
                            "[%d] The task %s was placed before a running parent task "
                            "%s (with a remaining time of %s) finishes at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            parent.unique_name,
                            parent.remaining_time,
                            sim_time + parent.remaining_time,
                        )
                    else:
                        any_parent_placed = True
                elif parent.state == TaskState.COMPLETED:
                    # If the task was finished, this task should be good since its
                    # lower bound was checked correctly.
                    any_parent_placed = True
                else:
                    raise ValueError(
                        f"The placement for {placement.task.unique_name} occurred "
                        f"even though its parent {parent.unique_name} was in the "
                        f"{parent.state} state."
                    )

            if placement.task.terminal and not any_parent_placed:
                self._logger.error(
                    "[%s] The Task %s was a terminal task that was placed "
                    "without placing any of its parents: %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    placement.task.unique_name,
                    [parent.unique_name for parent in parents],
                )

        def check_for_oversubscription(
            task: Task,
            task_placed_on_worker_pool: WorkerPool,
            overlapping_tasks: Sequence[Task],
        ):
            # For all the resources that are required by the task, ensure that they
            # are not oversubscribed by the overlapping tasks.
            for (
                resource,
                quantity,
            ) in task.resource_requirements.get_unique_resource_types().items():
                overlapping_task_requirements_for_resource = [
                    (
                        overlapping_task.unique_name,
                        overlapping_task.resource_requirements.get_total_quantity(
                            resource
                        ),
                    )
                    for overlapping_task in overlapping_tasks
                ]
                if sum(
                    map(itemgetter(1), overlapping_task_requirements_for_resource)
                ) + quantity > task_placed_on_worker_pool.resources.get_total_quantity(
                    resource
                ):
                    self._logger.error(
                        "[%s] The task %s in state %s which was placed on %s and "
                        "required %s units of %s led to the oversubscription of the "
                        "%s quantities of resource at the WorkerPool because its "
                        "overlapping tasks had a cumulative requirement of %s with "
                        "the following individual requirements for the resource: %s.",
                        sim_time.to(EventTime.Unit.US).time,
                        task.unique_name,
                        task.state,
                        task_placed_on_worker_pool.name,
                        quantity,
                        resource,
                        task_placed_on_worker_pool.resources.get_total_quantity(
                            resource
                        ),
                        sum(
                            map(
                                itemgetter(1),
                                overlapping_task_requirements_for_resource,
                            )
                        ),
                        overlapping_task_requirements_for_resource,
                    )
                    raise ValueError(
                        f"The Placement for {task.unique_name} led to "
                        f"oversubscription of {resource.name} at "
                        f"{task_placed_on_worker_pool.name}."
                    )

        # Find all the required information about all the RUNNING, SCHEDULED and tasks
        # that were placed in this iteration of the Scheduler.
        task_placements = []
        for running_task in workload.filter(lambda t: t.state == TaskState.RUNNING):
            task_placements.append(
                (
                    # A reference to the Task object.
                    running_task,
                    # The time at which the Task is expected to start.
                    sim_time,
                    # The ID of the WorkerPool where the Task was placed.
                    running_task.worker_pool_id,
                )
            )
        for scheduled_task in workload.filter(lambda t: t.state == TaskState.SCHEDULED):
            # TODO (Sukrit): If a previously scheduled task is rescheduled, use the
            # latest placement.
            task_placements.append(
                (
                    scheduled_task,
                    scheduled_task.current_placement.placement_time,
                    scheduled_task.current_placement.worker_pool_id,
                )
            )
        for placement in placements:
            if not placement.is_placed():
                continue
            task_placements.append(
                (
                    placement.task,
                    placement.placement_time,
                    placement.worker_pool_id,
                )
            )

        # For all the tasks that are to be placed before, ensure that their
        # placement does not oversubscribe the available resources.
        for scheduled_task in workload.filter(lambda t: t.state == TaskState.SCHEDULED):
            # For previously scheduled tasks, ensure that the new placements along with
            # any previously running tasks don't oversubscribe the resources.
            overlapping_tasks = []
            for task, expected_start_time, worker_pool_id in task_placements:
                if (
                    task.id != scheduled_task.id
                    and worker_pool_id
                    == scheduled_task.current_placement.worker_pool_id
                    and expected_start_time <= scheduled_task.expected_start_time
                    and expected_start_time + task.remaining_time
                    >= scheduled_task.expected_start_time
                ):
                    overlapping_tasks.append(task)

            # Check for oversubscription.
            task_placed_on_worker_pool = worker_pools.get_worker_pool(
                scheduled_task.current_placement.worker_pool_id
            )
            if task_placed_on_worker_pool is None:
                raise ValueError(
                    f"A WorkerPool with ID "
                    f"{scheduled_task.current_placement.worker_pool_id} was not found."
                )
            check_for_oversubscription(
                scheduled_task, task_placed_on_worker_pool, overlapping_tasks
            )

        # For all the tasks that were placed right now, ensure that their placement
        # did not violate any resource availability.
        for placement in placements:
            if not placement.is_placed():
                continue
            # For tasks scheduled in the current run, ensure that the running and
            # previously placed tasks do not oversubscribe the resources.
            overlapping_tasks = []
            for task, expected_start_time, worker_pool_id in task_placements:
                if (
                    task.id != placement.task.id
                    and worker_pool_id == placement.worker_pool_id
                    and expected_start_time <= placement.placement_time
                    and expected_start_time + task.remaining_time
                    >= placement.placement_time
                ):
                    overlapping_tasks.append(task)

            # Check for oversubscription.
            task_placed_on_worker_pool = worker_pools.get_worker_pool(
                placement.worker_pool_id
            )
            if task_placed_on_worker_pool is None:
                raise ValueError(
                    f"A WorkerPool with ID "
                    f"{scheduled_task.current_placement.worker_pool_id} was not found."
                )
            check_for_oversubscription(
                placement.task, task_placed_on_worker_pool, overlapping_tasks
            )
