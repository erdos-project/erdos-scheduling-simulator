import heapq
from copy import deepcopy
from typing import Optional

import absl  # noqa: F401

from utils import EventTime, setup_logging
from workers import WorkerPools
from workload import BranchPredictionPolicy, Placements, TaskState, Workload


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
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
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
        retract_schedules: bool = False,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        self._preemptive = preemptive
        self._runtime = runtime
        self._lookahead = lookahead
        self._enforce_deadlines = enforce_deadlines
        self._policy = policy
        self._retract_schedules = retract_schedules
        self._release_taskgraphs = _flags.release_taskgraphs if _flags else False
        self._flags = _flags

        if self._flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_file=self._flags.log_file_name,
                log_level=self._flags.log_level,
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)

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
                self._logger.warn(
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
                >= placement.task.deadline
            ):
                self._logger.warn(
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
                parent_placement = placements.get_placement(parent)
                if parent.state in (
                    TaskState.VIRTUAl,
                    TaskState.RELEASED,
                    TaskState.PREEMPTED,
                ) or (
                    parent.state == TaskState.SCHEDULED and parent_placement is not None
                ):
                    # If the parent was supposed to be scheduled by this run, ensure
                    # that all the dependencies are being satisfied.
                    if parent_placement is None or not parent_placement.is_placed():
                        self._logger.warn(
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

                    if (
                        parent_placement.placement_time + parent.remaining_time
                        <= placement.placement_time
                    ):
                        self._logger.warn(
                            "[%d] The task %s was placed at %s before the parent %s "
                            "(placed at %s with remaining time of %s) finishes at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            parent.unique_name,
                            parent_placement.placement_time,
                            parent.remaining_time,
                            parent_placement.placement_time + parent.remaining_time,
                        )
                        raise ValueError(
                            f"The placement for {placement.task.unique_name} did not "
                            f"respect the task dependencies."
                        )
                elif parent.state == TaskState.SCHEDULED and parent_placement is None:
                    # If the task was scheduled previously, and was not reconsidered
                    # for scheduling, use the previously decided start time.
                    if (
                        parent.expected_start_time + parent.remaining_time
                        <= placement.placement_time
                    ):
                        self._logger.warn(
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
                elif parent.state == TaskState.RUNNING:
                    # If the task is currently running, use the remaining time to
                    # ensure that the dependencies were respected.
                    if sim_time + parent.remaining_time <= placement.placement_time:
                        self._logger.warn(
                            "[%d] The task %s was placed before a running parent task "
                            "%s (with a remaining time of %s) finishes at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            parent.unique_name,
                            parent.remaining_time,
                            sim_time + parent.remaining_time,
                        )
                elif parent.state == TaskState.COMPLETED:
                    # If the task was finished, this task should be good since its
                    # lower bound was checked correctly.
                    pass
                else:
                    raise ValueError(
                        f"The placement for {placement.task.unique_name} occurred "
                        f"even though its parent {parent.unique_name} was in the "
                        f"{parent.state} state."
                    )

        # TODO (Sukrit): Ensure that the WorkerPools are not oversubscribed.
