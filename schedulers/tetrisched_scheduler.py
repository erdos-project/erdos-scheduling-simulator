import os
import time
from collections import defaultdict
from math import ceil
from typing import List, Mapping, Optional, Set, Tuple

import absl  # noqa: F401
import numpy as np
import tetrisched_py as tetrisched

from schedulers import BaseScheduler
from utils import EventTime, generate_monotonically_increasing_intervals
from workers import WorkerPools
from workload import (
    Placement,
    Placements,
    Resource,
    Task,
    TaskGraph,
    TaskState,
    Workload,
)


class Partitions(object):
    """A wrapper class over tetrisched.Partitions that provides helper methods to
    enable construction of task STRLs and the retrieval of workers corresponding
    to the ultimate placement.

    Args:
        worker_pools (`WorkerPools`): The `WorkerPools` to initialize the
            Partitions with.
    """

    def __init__(self, worker_pools: WorkerPools) -> None:
        self._available_partitions = tetrisched.Partitions()
        self._resource_name_to_partitions_map: Mapping[str, tetrisched.Partitions] = (
            defaultdict(tetrisched.Partitions)
        )
        # BUG (Sukrit): The worker_index_to_partition_map is being used to keep the
        # Partition objects live on the Python side so we can query the associatedWorker
        # and the associatedWorkerPool. Otherwise, pybind11 loses track of the objects
        # and the attributes are not accessible.
        self._worker_index_to_partition_map: Mapping[int, tetrisched.Partition] = {}
        self._worker_index_counter = 1

        # Construct the Partitions object from the WorkerPools.
        self._construct_partitions(worker_pools=worker_pools)

    def _construct_partitions(self, worker_pools: WorkerPools) -> None:
        """Partitions the Workers in the WorkerPools into a granular partition set.

        The Partitions are used to reduce the number of variables in the compiled ILP
        model. All the resources in the Partition are expected to belong to an
        equivalence set and are therefore interchangeable.

        Args:
            worker_pools (`WorkerPools`): The WorkerPools to be partitioned.
        """
        # TODO (Sukrit): This method constructs a separate partition for all the slots
        # in a Worker. This might not be the best strategy for dealing with heterogenous
        # resources. Fix.
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                if len(worker.resources.resources) > 1:
                    raise NotImplementedError(
                        f"Partition generation only works for 1 resource type per "
                        f"Worker. Worker {worker.name} has "
                        f"{len(worker.resources.resources)} resources."
                    )
                resource, quantity = next(iter(worker.resources.resources))

                # Check that the Worker only has Slot resources.
                if not resource.name.startswith("Slot"):
                    raise NotImplementedError(
                        f"TetrischedScheduler currently supports Slot resources. "
                        f"Worker {worker.name} has resource with name {resource.name}."
                    )

                # Create a tetrisched Partition.
                partition = tetrisched.Partition(
                    self._worker_index_counter, worker.name, quantity
                )
                self._resource_name_to_partitions_map[resource.name].addPartition(
                    partition
                )
                self._available_partitions.addPartition(partition)

                # Maintain the relevant mappings to transform it to a Placement.
                partition.associatedWorker = worker
                partition.associatedWorkerPool = worker_pool
                self._worker_index_to_partition_map[self._worker_index_counter] = (
                    partition
                )
                self._worker_index_counter += 1

    def get_partition_for_worker_id(
        self, worker_id: str
    ) -> Optional[tetrisched.Partition]:
        for partition in self._worker_index_to_partition_map.values():
            if partition.associatedWorker.id == worker_id:
                return partition
        return None

    def get_partition_for_partition_id(
        self, partition_id: int
    ) -> Optional[tetrisched.Partition]:
        if partition_id in self._worker_index_to_partition_map:
            return self._worker_index_to_partition_map[partition_id]
        return None

    def __getitem__(self, resource: Resource) -> Optional[tetrisched.Partitions]:
        # If the Resource exists, retun the Partitions else return None.
        if resource.name in self._resource_name_to_partitions_map:
            return self._resource_name_to_partitions_map[resource.name]
        return None

    def __len__(self) -> int:
        """Returns the number of C++ Partitions objects in this wrapper.

        The number of C++ Partitions are defined as the number of different resources
        for which the Partitions are being generated.
        """
        return len(self._resource_name_to_partitions_map)

    @property
    def partitions(self) -> tetrisched.Partitions:
        return self._available_partitions


class TetriSchedScheduler(BaseScheduler):
    """Implements a STRL-based, DAG-aware formulation for the Tetrisched backend.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        lookahead (`EventTime`): The scheduler will try to place tasks that are within
            the scheduling lookahead (in us) using estimated task release times.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
        release_taskgraphs (`bool`): If `True`, the scheduler releases the TaskGraphs
            that are ready for scheduling. Turning this to `False` turns the scheduler
            to a task-by-task scheduler that only schedules the available frontier of
            tasks.
        goal (`str`): The goal to use as the optimization objective.
        time_discretization (`EventTime`): The time discretization at which the
            scheduling decisions are made.
        plan_ahead (`EventTime`): The time in the future up to which the time
            discretizations are to be generated for possible placements. The default
            value sets it to the maximum deadline from the available tasks. If the
            plan_ahead is set to low values, and `drop_skipped_tasks` is set in the
            Simulator, then tasks may be dropped that could have otherwise been
            scheduled leading to lower goodput.
        log_to_file (`bool`): If `True`, the scheduler writes the Gurobi search
            log to files with the format "gurobi_{sim_time}.log".
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
        plan_ahead_no_consideration_gap (`EventTime`): The time gap after the current
            time for which plan ahead is not considered malleable. Any tasks placed
            during this time will not be reconsidered for scheduling in a particular
            run.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime.invalid(),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        goal: str = "max_goodput",
        time_discretization: EventTime = EventTime(1, EventTime.Unit.US),
        plan_ahead: EventTime = EventTime.invalid(),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
        adaptive_discretization: bool = False,
        max_time_discretization: EventTime = EventTime(5, EventTime.Unit.US),
        max_occupancy_threshold: float = 0.8,
        finer_discretization_at_prev_solution: bool = False,
        finer_discretization_window: EventTime = EventTime(5, EventTime.Unit.US),
        plan_ahead_no_consideration_gap: EventTime = EventTime.zero(),
    ):
        if preemptive:
            raise ValueError("TetrischedScheduler does not support preemption.")
        super(TetriSchedScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        # Values for output.
        self._log_to_file = log_to_file
        self._log_dir = _flags.log_dir if _flags else "./"
        self._log_times = set(map(int, _flags.scheduler_log_times)) if _flags else set()

        self._goal = goal
        self._time_discretization = time_discretization.to(EventTime.Unit.US)
        self._plan_ahead = plan_ahead.to(EventTime.Unit.US)

        # Values for STRL generation.
        self._use_windowed_choose = False
        self._dynamic_discretization = (
            "DYNAMIC_DISCRETIZATION_PASS" in self._flags.opt_passes
        )
        self._adaptive_discretization = adaptive_discretization
        if self._dynamic_discretization or self._adaptive_discretization:
            # Force windowed choose to be false since these passes don't work with
            # windowed choose.
            self._use_windowed_choose = False
        self._max_discretization = max_time_discretization.to(EventTime.Unit.US)

        # optimization passes config
        self._opt_configuration = tetrisched.OptimizationPassConfig()
        self._opt_configuration.minDiscretization = self._time_discretization.time
        self._opt_configuration.maxDiscretization = self._max_discretization.time
        self._opt_configuration.maxOccupancyThreshold = max_occupancy_threshold
        self._opt_configuration.finerDiscretizationAtPrevSolution = (
            finer_discretization_at_prev_solution
        )
        self._opt_configuration.finerDiscretizationWindow = (
            finer_discretization_window.to(EventTime.Unit.US).time
        )

        self._scheduler = tetrisched.Scheduler(
            self._time_discretization.time,
            tetrisched.backends.SolverBackendType.GUROBI,
            self._log_dir,
            self._opt_configuration,
        )

        self._use_task_graph_indicator_utility = self._goal == "max_goodput"
        self._previously_placed_reward_scale_factor = 1.0

        # add optimization passes
        if len(self._flags.opt_passes) == 0:
            self._logger.info("Running with no Optimization Passes.")
        for opt_pass in self._flags.opt_passes:
            self._logger.info(f"Enabling the Optimization Pass: {opt_pass}")
            self._scheduler.addOptimizationPass(
                tetrisched.OptimizationPassCategory.__members__[opt_pass]
            )

        self._selectively_choose_task_graphs_for_rescheduling = (
            _flags.scheduler_selective_rescheduling if _flags else False
        )
        self._selectively_choose_task_graphs_sample_size = (
            _flags.scheduler_selective_rescheduling_sample_size if _flags else 5
        )
        self._plan_ahead_multiplier: int = 2
        self._plan_ahead_no_consideration_gap = plan_ahead_no_consideration_gap

        # Scheduler configuration.
        self._scheduler_configuration = tetrisched.SchedulerConfig()
        self._scheduler_configuration.totalSolverTimeMs = (
            # 1 minute interrupt by default.
            1 * 60 * 1000
            if _flags is None or _flags.scheduler_time_limit == -1
            else _flags.scheduler_time_limit * 1000
        )

        # NOTE (Sukrit): We observe that solving each TaskGraph independently usually
        # leads to more missed deadlines than required. To offset this, the following
        # parameter sets a threshold until which the scheduler will try to reschedule
        # the TaskGraphs in the next invocation of the scheduler.
        # For example, a value of 0.2 means that the scheduler will try to reschedule
        # the TaskGraphs until the scheduler invocations within 20% of the time between
        # the release time and the deadline. So, if a TaskGraph was released at 100 and
        # has a deadline of 500, it will be retried until scheduler invocations upto
        # 180, and will be dropped after.
        self._task_graph_reconsideration_period = (
            0.10 if _flags is None else _flags.scheduler_reconsideration_period
        )
        self._previously_considered_task_graphs: Set[str] = set()

        # A cache for the STRLs generated for individual tasks.
        # This is used to avoid generating the same STRL multiple times, and so that
        # we can add AllocationExpressions as children of ObjectiveExpressions to make
        # sure that no optimization passes remove it.
        # Reset at the beginning of each Scheduler invocation (`schedule()`).
        self._individual_task_strls: Mapping[str, tetrisched.Expression] = {}
        self._skipped_task_names: Set[str] = set()
        # A cache for previously selected choose expressions, this is done for
        # providing better hints to the solver
        self._previously_satisfied_choose_exprs = set([])
        if (
            self._adaptive_discretization
            and self._max_discretization.time < self._time_discretization.time
        ):
            raise ValueError(
                "Max dicretization should be greater than or equal to "
                "time discretization but currently it is not"
            )
        if self._adaptive_discretization and self._use_windowed_choose:
            raise ValueError(
                "Adaptive discretization and windowed choose cannot be used together."
            )
        if self._dynamic_discretization and self._use_windowed_choose:
            raise ValueError(
                "Dynamic discretization and windowed choose cannot be used together."
            )
        if self._adaptive_discretization and self._dynamic_discretization:
            raise ValueError(
                "Adaptive and Dynamic discretization cannot be used together."
            )
        if self._dynamic_discretization and self._time_discretization.time != 1:
            raise ValueError(
                "Dynamic discretization cannot be used with min Discretization > 1."
            )
        if self._goal != "max_goodput" and self._use_windowed_choose:
            raise NotImplementedError(
                f"Windowed choose not implemented for the goal: {self._goal}."
            )

    def _cancel_task_graph_predicate(self, task_graph: TaskGraph) -> bool:
        """Returns True if the TaskGraph should be skipped from scheduling."""
        return False

    def _cancel_task_graphs(
        self, current_time: EventTime, task_graph_names: Set[str], workload: Workload
    ) -> Set[str]:
        """Cancels the appropriate TaskGraphs from the given set and returns the names
        of the cancelled TaskGraphs.

        The decision of when to cancel a TaskGraph is made by either the predicate
        `_cancel_task_graph_predicate` or by the time until the TaskGraph was supposed
        to be reconsidred for scheduling. To alter the behavior of the latter, the
        `_task_graph_reconsideration_period` parameter can be used.

        Args:
            current_time (`EventTime`): The current time in the simulation.
            task_graph_names (`Set[str]`): The names of the TaskGraphs available for
                scheduling in this cycle.
            workload (`Workload`): The Workload to be used to retrieve the TaskGraphs.

        Returns:
            `Set[str]`: The names of the TaskGraphs that were cancelled.
        """
        cancelled_task_graphs: Set[str] = set()
        for task_graph_name in task_graph_names:
            # Retrieve the TaskGraph.
            task_graph = workload.get_task_graph(task_graph_name)
            if task_graph is None:
                raise ValueError(
                    f"Could not find TaskGraph with name {task_graph_name}."
                )

            # If we are enforcing deadlines and there is no way to meet this TaskGraph's
            # deadline, then we cancel it upfront.
            if (
                self.enforce_deadlines
                and current_time + task_graph.get_remaining_time() > task_graph.deadline
            ):
                self._logger.debug(
                    f"[{current_time.time}] Cancelling TaskGraph {task_graph_name} "
                    f"since the remaining time is {task_graph.get_remaining_time()} "
                    f"and the deadline is {task_graph.deadline}. The earliest it "
                    f"could finish is {current_time + task_graph.get_remaining_time()}"
                )
                cancelled_task_graphs.add(task_graph_name)
                continue

            # TaskGraphs that have been previously scheduled cannot be cancelled
            # upfront since they already have a feasible placement. The scheduler
            # must choose to cancel or keep them later.
            if task_graph.is_scheduled():
                # The TaskGraph has been scheduled before. Keep it.
                self._logger.debug(
                    f"[{current_time.time}] Keeping TaskGraph {task_graph_name} "
                    f"released at {task_graph.release_time} with deadline "
                    f"{task_graph.deadline} since it has been scheduled before."
                )
                continue

            # Check if the TaskGraph needs to be cancelled.
            if task_graph_name not in self._previously_considered_task_graphs:
                # If this is a new TaskGraph, check if it needs to be cancelled
                # upfront as decided by the predicate
                if self._cancel_task_graph_predicate(task_graph):
                    # If the predicate says that we should cancel the TaskGraph
                    # without trying, we just add the TaskGraph to the set of cancelled
                    # TaskGraphs.
                    self._logger.debug(
                        f"[{current_time.time}] Cancelling TaskGraph {task_graph_name} "
                        f"since the predicate that decides whether to cancel it is "
                        f"True."
                    )
                    self._previously_considered_task_graphs.add(task_graph_name)
                    cancelled_task_graphs.add(task_graph_name)
            else:
                # The TaskGraph has not been scheduled before, and is being
                # reconsidered for scheduling. Calculate the slack between the
                # release time and the deadline and decide whether the TaskGraph
                # should be dropped.
                slack = task_graph.deadline - task_graph.release_time
                time_until_reconsideration_ends = task_graph.release_time + EventTime(
                    ceil(slack.time * self._task_graph_reconsideration_period),
                    EventTime.Unit.US,
                )
                if current_time > time_until_reconsideration_ends:
                    # The TaskGraph has been reconsidered for too long. Cancel it.
                    self._logger.debug(
                        f"[{current_time.time}] Cancelling TaskGraph {task_graph_name} "
                        f"because it has been reconsidered for too long. It was "
                        f"released at {task_graph.release_time} with deadline "
                        f"{task_graph.deadline}, and was reconsidered until "
                        f"{time_until_reconsideration_ends}."
                    )
                    cancelled_task_graphs.add(task_graph_name)
                else:
                    # The TaskGraph has not been reconsidered for too long. Keep it.
                    self._logger.debug(
                        f"[{current_time.time}] Keeping TaskGraph {task_graph_name} "
                        f"released at {task_graph.release_time} with deadline "
                        f"{task_graph.deadline}, and it will be reconsidered until "
                        f"{time_until_reconsideration_ends}."
                    )
        return cancelled_task_graphs

    def _get_plan_ahead_this_cycle(
        self,
        current_time: EventTime,
        workload: Workload,
        tasks: List[Task],
        task_graph_names: Set[str],
        cancelled_task_graphs: Set[str],
    ) -> EventTime:
        """Returns the plan-ahead for this scheduling cycle.

        Args:
            current_time (`EventTime`): The current time in the simulation.
            workload (`Workload`): The Workload to be used to retrieve the TaskGraphs.
            tasks (`List[Task]`): The tasks to be scheduled in this cycle.
            task_graph_names (`Set[str]`): The names of the TaskGraphs available for
                scheduling in this cycle.
            cancelled_task_graphs (`Set[str]`): The names of the TaskGraphs that were
                cancelled upfront.

        Returns:
            `EventTime`: The plan-ahead for this scheduling cycle.
        """
        plan_ahead_this_cycle = None
        if self.enforce_deadlines and self._plan_ahead.is_invalid():
            plan_ahead_this_cycle = max(task.deadline for task in tasks)
        else:
            if self._plan_ahead.is_invalid():
                # If no plan-ahead was provided, we use a configurable multiple of the
                # maximum remaining time across all the schedulable TaskGraphs.
                # Earlier iterations of this code used the sum of the remainder of the
                # critical paths for each of the TaskGraphs available to the scheduler
                # for this cycle. However, this led to an extraordinary amount of time
                # being stuck in presolve, and not actually solving the model using
                # branch and bound. We did not see any significant difference in the
                # goodput using this approach. For cases where the TaskGraphs are
                # not released, we use the sum of the remaining runtimes of all
                # the available tasks for scheduling.
                if self.release_taskgraphs:
                    plan_ahead = EventTime.zero()
                    for task_graph_name in task_graph_names:
                        if task_graph_name not in cancelled_task_graphs:
                            task_graph = workload.get_task_graph(task_graph_name)
                            if task_graph is None:
                                raise ValueError(
                                    f"Could not find TaskGraph with name "
                                    f"{task_graph_name}."
                                )
                            plan_ahead = max(
                                task_graph.get_remaining_time(), plan_ahead
                            )
                    plan_ahead = plan_ahead * self._plan_ahead_multiplier
                    self._logger.debug(
                        "[%s] The plan-ahead for this cycle was computed based on the "
                        "remaining time of the TaskGraphs and is %s.",
                        current_time.time,
                        plan_ahead,
                    )
                else:
                    plan_ahead = sum(
                        (task.remaining_time for task in tasks),
                        start=EventTime.zero(),
                    )
                    self._logger.debug(
                        "[%s] The plan-ahead for this cycle was computed based on "
                        "the runtimes of the schedulable tasks and is %s.",
                        current_time.time,
                        plan_ahead,
                    )
                plan_ahead_this_cycle = current_time + plan_ahead
            else:
                plan_ahead_this_cycle = current_time + self._plan_ahead
        return plan_ahead_this_cycle

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Reset the STRL mappings.
        self._individual_task_strls = {}

        # The Placements to be returned.
        placements = []

        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled: List[Task] = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
            release_taskgraphs=self.release_taskgraphs,
        )

        task_description_string = [
            f"{t.unique_name} ("
            f"{t.available_execution_strategies.get_fastest_strategy().runtime}, "
            f"{t.deadline})"
            for t in tasks_to_be_scheduled
        ]
        task_graph_names: Set[str] = {task.task_graph for task in tasks_to_be_scheduled}
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks to be scheduled from {len(task_graph_names)} TaskGraphs. "
            f"These tasks along with their "
            f"(runtimes, deadlines) were: {task_description_string}."
        )

        # Find the TaskGraphs that are past their reconsideration deadline and cancel
        # those upfront.
        cancelled_task_graphs: Set[str] = set()
        if self.release_taskgraphs and self.enforce_deadlines:
            cancelled_task_graphs = self._cancel_task_graphs(
                current_time=sim_time,
                task_graph_names=task_graph_names,
                workload=workload,
            )

            # Find the tasks that belong to any of these TaskGraphs and emit a
            # TASK_CANCEL event for each one of them.
            for task in tasks_to_be_scheduled:
                if task.task_graph in cancelled_task_graphs:
                    placements.append(Placement.create_task_cancellation(task=task))
        elif self.enforce_deadlines:
            # If we are not releasing TaskGraphs, then we just cancel the tasks that
            # cannot be finished by their deadlines.
            for task in tasks_to_be_scheduled:
                fastest_strategy = (
                    task.available_execution_strategies.get_fastest_strategy()
                )
                if task.deadline < sim_time + fastest_strategy.runtime:
                    placements.append(Placement.create_task_cancellation(task=task))
                    cancelled_task_graphs.add(task.task_graph)
                    self._logger.debug(
                        "[%s] Cancelled Task %s belonging to TaskGraph %s since its "
                        "deadline is at %s, and the fastest strategy can finish it in "
                        "%s.",
                        sim_time.time,
                        task.unique_name,
                        task.task_graph,
                        task.deadline,
                        fastest_strategy.runtime,
                    )

        # Find the currently running and scheduled tasks to inform
        # the scheduler of previous placements.
        if self.retract_schedules:
            # If we are retracting schedules, the scheduler will re-place
            # the scheduled tasks, so we should only consider RUNNING tasks.
            filter_fn = lambda task: task.state == TaskState.RUNNING  # noqa: E731
        else:
            # If we are not retracting schedules, we should consider both
            # RUNNING and SCHEDULED task placements as permanent.
            filter_fn = lambda task: task.state in (  # noqa: E731
                TaskState.RUNNING,
                TaskState.SCHEDULED,
            )
        previously_placed_tasks = workload.filter(filter_fn)
        self._logger.debug(
            f"[{sim_time.time}] The scheduler is also considering the following "
            f"{len(previously_placed_tasks)} for their effects on the current "
            f"placements: {[task.unique_name for task in previously_placed_tasks]}."
        )
        previously_placed_task_graphs: Set[str] = {
            task.task_graph for task in previously_placed_tasks
        }
        self._logger.debug(
            f"[{sim_time.time}] The previously placed tasks being considered for their "
            f"effects on the current placements were from the following "
            f"{len(previously_placed_task_graphs)} TaskGraphs: "
            f"{previously_placed_task_graphs}."
        )

        # Construct the STRL expression.
        scheduler_start_time = time.time()
        if len(tasks_to_be_scheduled) > 0 and any(
            # If there is a Task belonging to a TaskGraph that hasn't
            # been cancelled, then we run the scheduler.
            task.state != TaskState.SCHEDULED
            and task.task_graph not in cancelled_task_graphs
            for task in tasks_to_be_scheduled
        ):
            # Construct the partitions from the Workers in the WorkerPool.
            partitions = Partitions(worker_pools=worker_pools)

            # Construct the ObjectiveExpression to be optimized.
            objective_strl = tetrisched.strl.ObjectiveExpression(
                f"TetriSched_{sim_time.to(EventTime.Unit.US).time}"
            )

            # Construct the rewards for placement of the tasks.
            # Find the plan-ahead window to normalize the rewards for the tasks.
            # If enforce_deadlines is set to true, then we use the maximum deadline
            # across all the jobs in this scheduling cycle to decide the plan-ahead.
            plan_ahead_this_cycle = self._get_plan_ahead_this_cycle(
                sim_time,
                workload,
                tasks_to_be_scheduled,
                task_graph_names,
                cancelled_task_graphs,
            )
            self._logger.debug(
                "[%s] The plan-ahead for this scheduling cycle was set to %s.",
                sim_time.time,
                plan_ahead_this_cycle,
            )

            if not self._adaptive_discretization:
                placement_reward_discretizations = self._get_time_discretizations_until(
                    current_time=sim_time, end_time=plan_ahead_this_cycle
                )
                start_end_time_list = []
            else:
                (
                    placement_reward_discretizations,
                    start_end_time_list,
                ) = self._get_time_discretizations_until(
                    current_time=sim_time,
                    end_time=plan_ahead_this_cycle,
                    return_start_end_times=True,
                )

            self._logger.debug(
                f"[{sim_time.time}] The plan ahead for this scheduler invocation was "
                f"{plan_ahead_this_cycle}, and the resulting discretizations were: "
                f"{[str(t) for t in placement_reward_discretizations]}."
            )

            # If the goal of the scheduler is to minimize the placement delay, we
            # value earlier placement choices for each task higher. Note that this
            # usually leads to a higher scheduler runtime since the solver has to close
            # the gap between the best bound and the objective.
            placement_times_and_rewards = None
            if self._goal == "min_placement_delay":
                placement_times_and_rewards = list(
                    zip(
                        placement_reward_discretizations,
                        np.interp(
                            list(
                                map(lambda x: x.time, placement_reward_discretizations)
                            ),
                            (
                                min(placement_reward_discretizations).time,
                                max(placement_reward_discretizations).time,
                            ),
                            (2, 1),
                        ),
                    )
                )
            else:
                # If the goal is not to minimize placement delay, we
                # value all the slots equivalently.
                placement_times_and_rewards = [
                    (t, 1) for t in placement_reward_discretizations
                ]

            # Keep track of the Tasks that have not been considered for scheduling.
            self._skipped_task_names: Set[str] = set()
            if self.release_taskgraphs:
                # Find the TaskGraphs that are available for scheduling.
                task_graphs_for_scheduling = self._choose_task_graphs_for_scheduling(
                    sim_time, workload, task_graph_names
                )
                self._logger.debug(
                    "[%s] A total of %s TaskGraphs were chosen for scheduling out of "
                    "%s. These were: %s",
                    sim_time.time,
                    len(task_graphs_for_scheduling),
                    len(task_graph_names),
                    task_graphs_for_scheduling,
                )

                # Construct the STRL expressions for each TaskGraph and add them
                # together in a single objective expression.
                task_strls: Mapping[str, tetrisched.strl.Expression] = {}
                for task_graph_name in task_graph_names:
                    # Add the TaskGraph to the previously considered TaskGraphs.
                    self._previously_considered_task_graphs.add(task_graph_name)

                    # Retrieve the TaskGraph and construct its STRL, if the TaskGraph
                    # hasn't already been cancelled.
                    if task_graph_name in cancelled_task_graphs:
                        self._logger.debug(
                            f"[{sim_time.time}] Skipping STRL generation for TaskGraph "
                            f"{task_graph_name} because it has been cancelled in this "
                            f"run."
                        )
                        continue

                    # Retrieve the TaskGraph.
                    task_graph = workload.get_task_graph(task_graph_name)

                    # If the TaskGraph has already been scheduled, then we see if we
                    # need to allow it to be rescheduled based on the given predicate.
                    if (
                        self._selectively_choose_task_graphs_for_rescheduling
                        and task_graph_name not in task_graphs_for_scheduling
                    ):
                        self._logger.debug(
                            f"[{sim_time.time}] Converting TaskGraph {task_graph_name} "
                            f"to an AllocationExpression because is_scheduled: "
                            f"{task_graph.is_scheduled()} "
                            f"and the predicate decided not to reschedule it."
                        )

                        # This TaskGraph is not to be rescheduled. Just add Allocation
                        # Expressions for the tasks that have already been placed.
                        for task in tasks_to_be_scheduled:
                            if task.task_graph == task_graph_name:
                                task_strl = self.construct_task_strl(
                                    sim_time,
                                    task,
                                    partitions,
                                    placement_times_and_rewards,
                                    retract_schedules=False,
                                )
                                if task_strl is not None:
                                    objective_strl.addChild(task_strl)
                                else:
                                    raise RuntimeError(
                                        f"Could not construct STRL for Task "
                                        f"{task.unique_name}. This is required for "
                                        f"previously placed tasks to account for "
                                        f"correct Allocations."
                                    )
                                self._skipped_task_names.add(task.unique_name)
                                self._logger.debug(
                                    "[%s] Adding %s to the tasks being "
                                    "skipped for scheduling.",
                                    sim_time.time,
                                    task.unique_name,
                                )
                        continue

                    # Construct the STRL.
                    scale_factor = self._previously_placed_reward_scale_factor
                    task_graph_strl = self.construct_task_graph_strl(
                        current_time=sim_time,
                        task_graph=task_graph,
                        partitions=partitions,
                        placement_times_and_rewards=placement_times_and_rewards,
                        tasks_to_be_scheduled=tasks_to_be_scheduled
                        + previously_placed_tasks,
                        task_strls=task_strls,
                        previously_placed=task_graph_name
                        in previously_placed_task_graphs,
                        use_indicator_utility=self._use_task_graph_indicator_utility,
                        scale_reward_previously_placed=scale_factor,
                    )
                    if task_graph_strl is not None:
                        objective_strl.addChild(task_graph_strl)

                # For the tasks that have been previously placed, add an
                # AllocationExpression for their current allocations so as to correctly
                # account for capacities at each time discretization.
                for task in previously_placed_tasks:
                    # If this child is not in the TaskGraphs to be scheduled, then we
                    # add it to the root expression.
                    task_strl = None
                    if task.id not in self._individual_task_strls:
                        task_strl = self.construct_task_strl(
                            sim_time,
                            task,
                            partitions,
                            placement_times_and_rewards,
                        )
                    else:
                        task_strl = self._individual_task_strls[task.id]
                        self._logger.debug(
                            f"[{sim_time.time}] Found STRL for Task "
                            f"{task.unique_name} with name {task_strl.name}."
                        )

                    if task_strl is not None:
                        objective_strl.addChild(task_strl)
                        self._logger.debug(
                            f"[{sim_time.time}] Added STRL for task {task.unique_name} "
                            f"to ObjectiveExpression with name {objective_strl.name}."
                        )
                    else:
                        raise RuntimeError(
                            f"Could not construct STRL for Task {task.unique_name}. "
                            f"This is required for previously placed tasks to account "
                            f"for correct Allocations."
                        )

            else:
                # If we are not releasing TaskGraphs, then we just construct the STRL
                # for the tasks that are to be scheduled.
                for task in tasks_to_be_scheduled + previously_placed_tasks:
                    task_strl = self.construct_task_strl(
                        sim_time,
                        task,
                        partitions,
                        placement_times_and_rewards,
                    )
                    if task_strl is not None:
                        objective_strl.addChild(task_strl)

            self._logger.debug(
                "[%s] The tasks being skipped from scheduling were: %s.",
                sim_time.time,
                ", ".join(list(self._skipped_task_names)),
            )

            # Register the STRL expression with the scheduler and solve it.
            try:
                self._scheduler.registerSTRL(
                    objective_strl,
                    partitions.partitions,
                    sim_time.time,
                    self._scheduler_configuration,
                    start_end_time_list,
                )
                solver_start_time = time.time()
                self._scheduler.schedule(sim_time.time)
                solver_end_time = time.time()
                solver_time = EventTime(
                    int((solver_end_time - solver_start_time) * 1e6), EventTime.Unit.US
                )
            except RuntimeError as e:
                self._logger.error(
                    f'[{sim_time.time}] Received error with description: "{e}" '
                    f"while invoking the STRL-based Scheduler. Dumping the model to "
                    f"tetrisched_error_{sim_time.time}.lp and STRL expression to "
                    f"tetrisched_error_{sim_time.time}.dot."
                )
                objective_strl.exportToDot(
                    os.path.join(self._log_dir, f"tetrisched_error_{sim_time.time}.dot")
                )
                self._scheduler.exportLastSolverModel(
                    os.path.join(self._log_dir, f"tetrisched_error_{sim_time.time}.lp")
                )
                raise e

            # If the solver could not solve the model, log an error.
            if not self._scheduler.getLastSolverSolution().isValid():
                self._logger.error(
                    f"[{sim_time.time}] The solver failed to find a solution for "
                    f"the STRL expression. Dumping the model to "
                    f"tetrisched_error_{sim_time.time}.lp and STRL expression to "
                    f"tetrisched_error_{sim_time.time}.dot."
                )
                objective_strl.exportToDot(
                    os.path.join(self._log_dir, f"tetrisched_error_{sim_time.time}.dot")
                )
                self._scheduler.exportLastSolverModel(
                    os.path.join(self._log_dir, f"tetrisched_error_{sim_time.time}.lp")
                )

            # If requested, log the model to a file.
            if self._log_to_file or sim_time.time in self._log_times:
                self._scheduler.exportLastSolverModel(
                    os.path.join(self._log_dir, f"tetrisched_{sim_time.time}.lp")
                )
                objective_strl.exportToDot(
                    os.path.join(self._log_dir, f"tetrisched_{sim_time.time}.dot")
                )
                self._logger.debug(
                    f"[{sim_time.to(EventTime.Unit.US).time}] Exported model to "
                    f"tetrisched_{sim_time.time}.lp and STRL to "
                    f"tetrisched_{sim_time.time}.dot"
                )

            # Retrieve the solution and check if we were able to schedule anything.
            solverSolution = objective_strl.getSolution()

            if solverSolution is not None and solverSolution.utility > 0:
                self._previously_satisfied_choose_exprs = (
                    solverSolution.satsifiedExpressionNames
                )
                self._logger.info(
                    f"The satisfied Choose Expressions: "
                    f"{self._previously_satisfied_choose_exprs}"
                )
                self._logger.info(
                    f"[{sim_time.time}] Solver returned utility of "
                    f"{solverSolution.utility} and took {solver_time} to solve. The "
                    f"solution result was {self._scheduler.getLastSolverSolution()}."
                )

                # Retrieve the Placements for each task.
                for task in tasks_to_be_scheduled:
                    if task.unique_name in self._skipped_task_names:
                        # If the task was skipped, then we honor the previous placement.
                        self._logger.debug(
                            f"[{sim_time.time}] Honoring prior placement for Task "
                            f"{task.unique_name} because it was skipped."
                        )
                        continue

                    task_placement = solverSolution.getPlacement(task.unique_name)
                    if task_placement is None or not task_placement.isPlaced():
                        self._logger.error(
                            f"[{sim_time.time}] No Placement was found for "
                            f"Task {task.unique_name}."
                        )
                        placements.append(Placement.create_task_placement(task=task))
                        continue

                    # Retrieve the Partition where the task was placed.
                    # The task was placed, retrieve the Partition where the task
                    # was placed.
                    partitionAllocations = task_placement.getPartitionAllocations()
                    try:
                        partitionId = list(partitionAllocations.keys())[0]
                    except IndexError as e:
                        self._logger.error(
                            f"[{sim_time.time}] Received error {e} while trying "
                            f"to access {partitionAllocations}."
                        )
                        raise e
                    partition = partitions.get_partition_for_partition_id(partitionId)

                    # Find the strategy that fits this Worker.
                    placement_execution_strategy_for_this_task = None
                    for execution_strategy in task.available_execution_strategies:
                        if len(execution_strategy.resources) > 1:
                            raise NotImplementedError(
                                f"TetrischedScheduler does not support multiple "
                                f"resources per execution strategy. The execution "
                                f"strategy {execution_strategy} for "
                                f"{task.unique_name} requires "
                                f"{len(execution_strategy.resources)} resources."
                            )
                        resource, quantity = next(
                            iter(execution_strategy.resources.resources)
                        )
                        if (
                            partition.associatedWorker.resources.get_total_quantity(
                                resource
                            )
                            >= quantity
                        ):
                            placement_execution_strategy_for_this_task = (
                                execution_strategy
                            )
                            break

                    if placement_execution_strategy_for_this_task is None:
                        raise RuntimeError(
                            f"[{sim_time.time}] Could not find a valid placement "
                            f"execution strategy for Task {task.unique_name} on "
                            f"Partition {partition.id}."
                        )

                    task_placement = Placement.create_task_placement(
                        task=task,
                        placement_time=EventTime(
                            task_placement.startTime, EventTime.Unit.US
                        ),
                        worker_id=partition.associatedWorker.id,
                        worker_pool_id=partition.associatedWorkerPool.id,
                        execution_strategy=placement_execution_strategy_for_this_task,
                    )
                    placements.append(task_placement)
                    self._logger.debug(
                        "[%s] Placed %s (with deadline %s and remaining time %s) on "
                        "WorkerPool (%s) to be started at %s and executed with %s.",
                        sim_time.to(EventTime.Unit.US).time,
                        task_placement.task.unique_name,
                        task_placement.task.deadline,
                        task_placement.execution_strategy.runtime,
                        task_placement.worker_pool_id,
                        task_placement.placement_time,
                        task_placement.execution_strategy,
                    )
            else:
                # There were no Placements from the Scheduler. Inform the Simulator.
                self._previously_satisfied_choose_exprs = set([])
                for task in tasks_to_be_scheduled:
                    placements.append(
                        Placement.create_task_placement(
                            task=task,
                            placement_time=None,
                            worker_pool_id=None,
                            worker_id=None,
                            execution_strategy=None,
                        )
                    )
                # There were no Placements from the Scheduler. We just skip the
                # placement of any of the tasks, and wait for the next invocation.
                self._logger.warning(f"[{sim_time.time}] Failed to place any tasks.")

        # if sim_time == EventTime(210, EventTime.Unit.US):
        #     raise RuntimeError("Stopping the Simulation.")

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime if self.runtime == EventTime.invalid() else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )

    def _get_time_discretizations_until(
        self, current_time: EventTime, end_time: EventTime, return_start_end_times=False
    ) -> List[EventTime]:
        """Constructs the time discretizations from current_time to end_time in the
        granularity provided by the scheduler.

        Note that the first time discretization is always <= current_time and should
        only be allowed placement for tasks in RUNNING state. This is because the
        simulator does not allow scheduling of tasks in the past.

        Args:
            current_time (`EventTime`): The time at which the scheduling is occurring.
            end_time (`EventTime`): The time at which the scheduling is to end.
            return_start_end_times(bool): Returns the list of start and end time
                with granularities

        Returns:
            A list of EventTimes that represent the time discretizations, and
            optionally start end times.
        """
        time_discretization = self._time_discretization.to(EventTime.Unit.US).time
        start_time = (
            (current_time.to(EventTime.Unit.US).time) // time_discretization
        ) * time_discretization
        end_time = end_time.to(EventTime.Unit.US).time

        discretizations = []
        start_end_times = []
        if not self._adaptive_discretization:
            for discretization_time in range(
                start_time, end_time + 1, time_discretization
            ):
                discretizations.append(
                    EventTime(discretization_time, EventTime.Unit.US)
                )
            if return_start_end_times:
                start_end_times.append(
                    ((start_time, end_time + 1), time_discretization)
                )
        else:
            min_discretization = self._time_discretization.to(EventTime.Unit.US).time
            max_discretization = self._max_discretization.to(EventTime.Unit.US).time
            num_interval = self._max_discretization.to(EventTime.Unit.US).time
            initial_repetitions = (
                end_time - start_time
            ) // 4  # 1/4th of the time min discretization should be repeated
            initial_repetitions = max(initial_repetitions, 1)

            intervals = generate_monotonically_increasing_intervals(
                min_discretization,
                max_discretization,
                num_interval,
                initial_repetitions,
            )
            total_intervals = len(intervals)
            interval_index = 0
            current_time = start_time
            while current_time < (end_time + 1):
                interval = intervals[min(interval_index, total_intervals - 1)]
                discretizations.append(EventTime(current_time, EventTime.Unit.US))
                if (current_time + interval) < (end_time + 1):
                    if return_start_end_times:
                        start_end_times.append(
                            (
                                (current_time, current_time + interval),
                                min(end_time - current_time + 1, interval),
                            )
                        )
                else:
                    if return_start_end_times:
                        start_end_times.append(
                            (
                                (current_time, end_time + 1),
                                min(end_time - current_time + 1, interval),
                            )
                        )

                current_time += interval

                interval_index += 1
        if not return_start_end_times:
            return discretizations
        return discretizations, start_end_times

    def construct_task_strl(
        self,
        current_time: EventTime,
        task: Task,
        partitions: Partitions,
        placement_times_and_rewards: List[Tuple[EventTime, float]],
        retract_schedules: Optional[bool] = None,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The Task for which the STRL expression is to be constructed.
            partitions (`Partitions`): The partitions that are available for scheduling.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices as ChooseExpressions and a MaxExpression that selects the best
            placement choice.
        """
        if len(placement_times_and_rewards) == 0:
            self._logger.debug(
                f"[{current_time.time}] No placement choices were available for "
                f"{task.unique_name} with deadline {task.deadline}."
            )
            self._individual_task_strls[task.id] = None
            return None

        # Set the retract_schedules flag.
        if retract_schedules is None:
            retract_schedules = self.retract_schedules

        self._logger.debug(
            "[%s] Constructing STRL for %s in state %s with retract_schedules %s.",
            current_time.time,
            task.unique_name,
            task.state,
            retract_schedules,
        )

        # We block off all the time discretizations from the task's start until it
        # completes if any one of the following conditions hold:
        # 1. The Task is already running (we do not consider preemption for now).
        # 2. The Task is scheduled and the scheduler was specifically requested to
        #    not retract schedules.
        # 3. The intended start time of the task falls within the no consideration
        #    period from the current time.
        if (
            task.state == TaskState.RUNNING  # 1.
            or (task.state == TaskState.SCHEDULED and not retract_schedules)  # 2.
            or (  # 3.
                task.state == TaskState.SCHEDULED
                and task.current_placement.placement_time
                < current_time + self._plan_ahead_no_consideration_gap
            )
        ):
            # Find the Partition where the task is running or to be scheduled.
            scheduled_partition = partitions.get_partition_for_worker_id(
                task.current_placement.worker_id
            )
            if scheduled_partition is None:
                raise ValueError(
                    f"Could not find the Partition for the Task "
                    f"{task.unique_name} in state {task.state}."
                )

            # Find the discretization where the Task is running or to be scheduled.
            scheduled_discretization = None
            if task.state == TaskState.RUNNING:
                scheduled_discretization, _ = placement_times_and_rewards[0]
                # BUG (Sukrit): If we go back in time and set the discretization from
                # the past, then we need to correctly account for the remaining time
                # from that point, instead of the current.
                task_remaining_time = (
                    current_time - scheduled_discretization
                ) + task.remaining_time
            else:
                for index, (time_discretization, _) in enumerate(
                    placement_times_and_rewards
                ):
                    if (
                        time_discretization <= task.current_placement.placement_time
                        and (
                            index == len(placement_times_and_rewards) - 1
                            or placement_times_and_rewards[index + 1][0]
                            > task.current_placement.placement_time
                        )
                    ):
                        # This is the first discretization where we should be blocking.
                        scheduled_discretization = time_discretization
                        break
                task_remaining_time = task.current_placement.execution_strategy.runtime

            if scheduled_discretization is None:
                raise ValueError(
                    f"Could not find the discretization for the Task "
                    f"{task.unique_name} in state {task.state} starting at "
                    f"{task.current_placement.placement_time} from "
                    f"{', '.join([str(t) for t, _ in placement_times_and_rewards])}."
                )

            # Find the number of resources required to execute this Task.
            if len(task.current_placement.execution_strategy.resources) > 1:
                raise ValueError(
                    f"{task.unique_name} in state {task.state} is being executed with "
                    f"{task.current_placement.execution_strategy} that requires "
                    f"{len(task.current_placement.execution_strategy.resources)} "
                    f"resources. TetrischedScheduler currently only supports 1 "
                    f"resource type per Task."
                )
            _, quantity = next(
                iter(task.current_placement.execution_strategy.resources.resources)
            )

            # Block off all the time discretizations from the task's start until it
            # completes.
            task_allocation_expression = tetrisched.strl.AllocationExpression(
                task.unique_name,
                [(scheduled_partition, quantity)],
                scheduled_discretization.to(EventTime.Unit.US).time,
                task_remaining_time.to(EventTime.Unit.US).time,
            )
            if task.state == TaskState.SCHEDULED:
                task_allocation_expression.setPreviouslySatisfied(True)

            self._skipped_task_names.add(task.unique_name)
            self._logger.debug(
                "[%s] Adding %s to the tasks being skipped for scheduling.",
                current_time.time,
                task.unique_name,
            )
            self._logger.debug(
                f"[{current_time.time}] Generated an AllocationExpression for "
                f"task {task.unique_name} in state {task.state} starting at "
                f"{scheduled_discretization} and running for {task.remaining_time} "
                f"on Partition {scheduled_partition.id}."
            )
            self._individual_task_strls[task.id] = task_allocation_expression
            return task_allocation_expression

        # If the task is not running or scheduled, we construct the STRL expression
        # for the task that makes it available for scheduling.
        task_execution_strategy_strls = []
        for execution_strategy in task.available_execution_strategies:
            self._logger.debug(
                f"[{current_time.time}] Considering strategy {execution_strategy} for "
                f"STRL generation for task {task.unique_name}."
            )
            # Find the partitions where this execution strategy is valid.
            if len(execution_strategy.resources) > 1:
                raise NotImplementedError(
                    f"TetrischedScheduler does not support multiple resources per "
                    f"execution strategy. The execution strategy {execution_strategy} "
                    f"for {task.unique_name} requires "
                    f"{len(execution_strategy.resources)} resources."
                )

            resource, quantity = next(iter(execution_strategy.resources.resources))
            if not resource.name.startswith("Slot"):
                raise NotImplementedError(
                    f"TetrischedScheduler only supports Slot resources. "
                    f"The resource {resource.name} is not supported."
                )
            partitions_for_this_execution_strategy = partitions[resource]
            if partitions_for_this_execution_strategy is None:
                self._logger.debug(
                    f"[{current_time.time}] No partitions for {resource.name} were "
                    f"available. Skipping STRL generation for strategy "
                    f"{execution_strategy} for task {task.unique_name}."
                )
                continue

            # Find the time at which we start enumerating the choices.
            time_until_choices_start = current_time.to(
                EventTime.Unit.US
            ) + self._plan_ahead_no_consideration_gap.to(EventTime.Unit.US)

            # Find the time until which we need to enumerate the choices.
            time_until_choices_end = placement_times_and_rewards[-1][0]
            if self.enforce_deadlines:
                time_until_choices_end = task.deadline - execution_strategy.runtime

            # Enumerate the time discretizations that are valid for this Task.
            time_discretizations = []
            for placement_time, reward in placement_times_and_rewards:
                if (
                    placement_time >= time_until_choices_start
                    and placement_time <= time_until_choices_end
                ):
                    time_discretizations.append((placement_time, reward))

            if len(time_discretizations) == 0:
                self._logger.warn(
                    f"[{current_time.time}] No time discretizations were feasible for "
                    f"{task.unique_name} from range {current_time} to "
                    f"{task.deadline - execution_strategy.runtime}."
                )
                continue

            # Validate that a hint was provided if required.
            was_hint_provided = False

            if not self._use_windowed_choose or len(time_discretizations) == 1:
                # We now construct the Choose expressions for each possible placement
                # choice of this Task, and collate them under a MaxExpression.
                task_choose_expressions = []
                choice_placement_times_and_rewards = []
                for placement_time, reward_for_this_placement in sorted(
                    time_discretizations
                ):
                    if placement_time < current_time:
                        # If the placement time is in the past, then we cannot
                        # place the task.
                        continue

                    # If the Task was scheduled before, and this time corresponds to
                    # the previous scheduling time, we provide a hint to the Scheduler
                    # so that it can reduce the overall solver time.
                    choose_expression_status = (
                        tetrisched.strl.ExpressionStatus.EXPR_STATUS_UNKNOWN
                    )
                    placement_hint = None
                    if task.state == TaskState.SCHEDULED:
                        choose_expression_status = (
                            tetrisched.strl.ExpressionStatus.EXPR_STATUS_UNSATISFIED
                        )
                        if (
                            task.current_placement.execution_strategy
                            == execution_strategy
                            and task.current_placement.placement_time == placement_time
                        ):
                            choose_expression_status = (
                                tetrisched.strl.ExpressionStatus.EXPR_STATUS_SATISFIED
                            )

                            # Retrieve the partition for the previous placement.
                            scheduled_partition = (
                                partitions.get_partition_for_worker_id(
                                    task.current_placement.worker_id
                                )
                            )
                            if scheduled_partition is None:
                                raise RuntimeError(
                                    f"Could not find the Partition for "
                                    f"the Task {task.unique_name} in state "
                                    f"{task.state} that was previously placed on "
                                    f"{task.current_placement.worker_id}."
                                )

                            placement_hint = [(scheduled_partition, quantity)]

                    if choose_expression_status == (
                        tetrisched.strl.ExpressionStatus.EXPR_STATUS_SATISFIED
                    ):
                        self._logger.debug(
                            "[%s] Hinting the placement of %s in state %s for time "
                            "%s with the previous placement %s.",
                            current_time.to(EventTime.Unit.US).time,
                            task.unique_name,
                            task.state,
                            placement_time,
                            placement_hint,
                        )
                        was_hint_provided = True

                    # Construct a ChooseExpression for placement at this time.
                    # TODO (Sukrit): We just assume for now that all Slots are the same
                    # and thus the task can be placed on any Slot. This is not true
                    # in general.
                    task_choose_expressions.append(
                        tetrisched.strl.ChooseExpression(
                            task.unique_name,
                            str(hash(execution_strategy)),
                            partitions_for_this_execution_strategy,
                            quantity,
                            placement_time.to(EventTime.Unit.US).time,
                            execution_strategy.runtime.to(EventTime.Unit.US).time,
                            reward_for_this_placement,
                            choose_expression_status,
                            placement_hint,
                        )
                    )
                    if (
                        task_choose_expressions[-1].discriptiveName
                        in self._previously_satisfied_choose_exprs
                    ):
                        task_choose_expressions[-1].setPreviouslySatisfied(True)
                    choice_placement_times_and_rewards.append(
                        (placement_time.time, reward_for_this_placement)
                    )

                if len(task_choose_expressions) == 0:
                    self._logger.warn(
                        f"[{current_time.time}] No ChooseExpressions were generated "
                        f"for {task.unique_name} with deadline {task.deadline}."
                    )
                elif len(task_choose_expressions) == 1:
                    self._logger.debug(
                        f"[{current_time.time}] Generated a single ChooseExpression "
                        f"for {task.unique_name} starting at "
                        f"{choice_placement_times_and_rewards[0][0]} "
                        f"with deadline {task.deadline} and a reward "
                        f"{choice_placement_times_and_rewards[0][1]}."
                    )
                    task_execution_strategy_strls.extend(task_choose_expressions)
                else:
                    if self._goal == "min_placement_delay":
                        self._logger.debug(
                            f"[{current_time.time}] Generated "
                            f"{len(choice_placement_times_and_rewards)} "
                            f"ChooseExpressions for {task.unique_name} for times and "
                            f"rewards {choice_placement_times_and_rewards} for "
                            f"{quantity} slots for {execution_strategy.runtime}."
                        )
                    else:
                        self._logger.debug(
                            f"[{current_time.time}] Generated "
                            f"{len(choice_placement_times_and_rewards)} "
                            f"ChooseExpressions for {task.unique_name} with deadline "
                            f"{task.deadline} and strategy {execution_strategy} for "
                            f"{quantity} slots for {execution_strategy.runtime}."
                        )
                    task_execution_strategy_strls.extend(task_choose_expressions)
            else:
                # We need to use a WindowedChoose here instead of generating Choose
                # expressions ourselves.
                start_time_discretization = (
                    time_discretizations[1][0].to(EventTime.Unit.US)
                    if time_discretizations[0][0].to(EventTime.Unit.US) > current_time
                    else time_discretizations[0][0].to(EventTime.Unit.US)
                )
                end_time_discretization = time_discretizations[-1][0].to(
                    EventTime.Unit.US
                )
                task_windowed_choose = tetrisched.strl.WindowedChooseExpression(
                    task.unique_name,
                    partitions_for_this_execution_strategy,
                    quantity,
                    start_time_discretization.time,
                    execution_strategy.runtime.to(EventTime.Unit.US).time,
                    end_time_discretization.time,
                    self._time_discretization.to(EventTime.Unit.US).time,
                    1,
                )
                self._logger.debug(
                    f"[{current_time.time}] Generated a WindowedChooseExpression for "
                    f"{task.unique_name} with deadline {task.deadline} and execution "
                    f"strategy {execution_strategy} for times between "
                    f"{start_time_discretization} and {end_time_discretization} "
                    f"for {quantity} slots for duration {execution_strategy.runtime}."
                )
                task_execution_strategy_strls.append(task_windowed_choose)

            if (
                task.state == TaskState.SCHEDULED
                and not was_hint_provided
                and task.current_placement.execution_strategy == execution_strategy
            ):
                self._logger.error(
                    "[%s] Task %s was in state %s with the prior placement "
                    "time %s, but no corresponding hint was provided. The "
                    "ChooseExpressions were generated for times: %s.",
                    current_time.to(EventTime.Unit.US).time,
                    task.unique_name,
                    task.state,
                    task.current_placement.placement_time,
                    [t.time for t, _ in time_discretizations],
                )
                raise RuntimeError(
                    f"No hint was provided for {task.unique_name}, but one "
                    f"was expected at {task.current_placement.placement_time}."
                )

        # Construct the STRL MAX expression for this Task.
        # This enforces the choice of only one placement for this Task.
        if len(task_execution_strategy_strls) == 0:
            self._logger.debug(
                f"[{current_time.time}] No STRL expressions were generated for "
                f"{task.unique_name} with deadline {task.deadline}."
            )
            self._individual_task_strls[task.id] = None
            return None
        elif len(task_execution_strategy_strls) == 1:
            self._logger.debug(
                f"[{current_time.time}] Generated a single STRL expression for "
                f"{task.unique_name} with deadline {task.deadline}. Returning the "
                f"expression of type {task_execution_strategy_strls[0].getType()}"
            )
            self._individual_task_strls[task.id] = task_execution_strategy_strls[0]
            return task_execution_strategy_strls[0]
        else:
            self._logger.debug(
                f"[{current_time.time}] Constructing a STRL expression tree for "
                f"{task.name} (runtime={execution_strategy.runtime}, "
                f"deadline={task.deadline}) with name: "
                f"{task.unique_name}_placement."
            )
            chooseOneFromSet = tetrisched.strl.MaxExpression(
                f"{task.unique_name}_placement"
            )
            for choose_expression in task_execution_strategy_strls:
                chooseOneFromSet.addChild(choose_expression)

            self._individual_task_strls[task.id] = chooseOneFromSet
            return chooseOneFromSet

    def _construct_task_graph_strl(
        self,
        current_time: EventTime,
        task: Task,
        task_graph: TaskGraph,
        partitions: Partitions,
        task_strls: Mapping[str, tetrisched.strl.Expression],
        placement_times_and_rewards: List[Tuple[EventTime, float]],
        tasks_to_be_scheduled: Optional[List[Task]] = None,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph starting at
        the specified Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The task in the TaskGraph for which the STRL expression is
                to be rooted at.
            partitions (`Partitions`): The partitions that are available for scheduling.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices for all the Tasks in the TaskGraph and enforces ordering amongst
            them.
        """
        # Check if we have already constructed the STRL for this Task, and return
        # the expression if we have.
        if task.id in task_strls:
            self._logger.debug(
                "[%s] Reusing STRL for Task %s.", current_time.time, task.unique_name
            )
            return task_strls[task.id]

        # Construct the STRL expression for this Task.
        if tasks_to_be_scheduled is None or (
            tasks_to_be_scheduled is not None and task in tasks_to_be_scheduled
        ):
            self._logger.debug(
                f"[{current_time.time}] Constructing the TaskGraph STRL for the "
                f"graph {task_graph.name} rooted at {task.unique_name}."
            )
            task_expression = self.construct_task_strl(
                current_time, task, partitions, placement_times_and_rewards
            )
            if not task_expression:
                self._logger.warn(
                    f"[{current_time.time}] Could not construct the STRL for "
                    f"Task {task.unique_name}. Failing the construction of STRL "
                    f"for the TaskGraph {task_graph.name} rooted at {task.unique_name}."
                )
                return None
        else:
            # If this Task is not in the set of Tasks that we are required to schedule,
            # then we just return a None expression.
            self._logger.debug(
                f"[{current_time.time}] Task {task.unique_name} in state {task.state} "
                f"is not in the set of tasks to be scheduled."
            )
            task_expression = None

        # Retrieve the STRL expressions for all the children of this Task.
        child_expressions = {}
        for child in task_graph.get_children(task):
            child_expression = self._construct_task_graph_strl(
                current_time,
                child,
                task_graph,
                partitions,
                task_strls,
                placement_times_and_rewards,
                tasks_to_be_scheduled,
            )
            if child_expression:
                child_expressions[child_expression.id] = child_expression
            elif child.state != TaskState.COMPLETED:
                self._logger.warn(
                    f"[{current_time.time}] Could not construct the STRL for all the "
                    f"children of {task.unique_name}. Failing the construction of STRL "
                    f"for the TaskGraph {task_graph.name} rooted at {task.unique_name}."
                )
                return None

        # If there are no children, cache and return the expression for this Task.
        if len(child_expressions) == 0:
            task_strls[task.id] = task_expression
            return task_expression

        # Construct the subtree for the children of this Task.
        if len(child_expressions) > 1:
            # If there are more than one children, then we need to ensure that all
            # of them are placed by collating them under a MinExpression.
            self._logger.debug(
                f"[{current_time.time}] Collating the children of {task.unique_name} "
                f"under a MinExpression {task.unique_name}_children for STRL of the "
                f"TaskGraph {task_graph.name} rooted at {task.unique_name}."
            )
            child_expression = tetrisched.strl.MinExpression(
                f"{task.unique_name}_children"
            )
            for child in child_expressions.values():
                child_expression.addChild(child)
        else:
            # If there is just one child, then we can just use that subtree.
            child_expression = next(iter(child_expressions.values()))

        # Construct a LessThanExpression to order the two trees.
        # If the current Task has to be scheduled, then we need to ensure that it
        # is scheduled before its children.
        if task_expression:
            self._logger.debug(
                f"[{current_time.time}] Ordering the STRL for {task.unique_name} and "
                f"its children "
                f"{[child.unique_name for child in task_graph.get_children(task)]}"
                f" under a LessThanExpression {task.unique_name}_less_than for "
                f"STRL of the TaskGraph {task_graph.name} rooted at {task.unique_name}."
            )
            task_graph_expression = tetrisched.strl.LessThanExpression(
                f"{task.unique_name}_less_than"
            )
            task_graph_expression.addChild(task_expression)
            task_graph_expression.addChild(child_expression)
        else:
            task_graph_expression = child_expression

        # Cache and return the expression for this Task.
        task_strls[task.id] = task_graph_expression
        return task_graph_expression

    def _choose_task_graphs_for_scheduling(
        self,
        current_time: EventTime,
        workload: Workload,
        task_graph_names: List[str],
    ) -> List[str]:
        """Find the TaskGraphs that are to be scheduled in this cycle.

        Args:
            current_time (`EventTime`): The time at which the scheduling is occurring.
            workload (`Workload`): The workload instance associated with this run of
                the scheduling cycle.
            task_graph_names (`List[str]`): The names of the TaskGraphs that are
                available for scheduling in this cycle.
        """
        reschedulable_task_graphs = []
        task_graphs_for_scheduling = []
        for task_graph_name in task_graph_names:
            # Find the TaskGraph.
            task_graph = workload.get_task_graph(task_graph_name)
            if task_graph is None:
                raise ValueError(
                    f"Could not find the TaskGraph {task_graph_name} in the workload."
                )

            # If the Taskgraph hasn't been previously scheduled, then it's always
            # available for scheduling.
            if not task_graph.is_scheduled():
                task_graphs_for_scheduling.append(task_graph_name)
                continue
            reschedulable_task_graphs.append(task_graph_name)

        # If we have less than the sample size, just consider all of them.
        if (
            len(reschedulable_task_graphs)
            <= self._selectively_choose_task_graphs_sample_size
        ):
            task_graphs_for_scheduling.extend(reschedulable_task_graphs)
            return task_graphs_for_scheduling

        # Strategy 1: We do not reconsider the TaskGraphs for scheduling that have
        # a very low slack between their release time and the deadline given their
        # critical path.
        # allowed_slack = 1.20 # 20% slack from critical path runtime.
        # should_reschedule = (
        #     task_graph.deadline.time
        #     >= task_graph.release_time.time
        #     + task_graph.critical_path_runtime.time * allowed_slack
        # )

        # Strategy 2: We try to only allow rescheduling of the TaskGraphs whose prior
        # placements of the sinks have a large slack between their deadline as a
        # percentage of their critical path runtime.
        # allowed_slack = 0.20  # 20% slack from critical path runtime.
        # completion_slack = min(
        #     [
        #         task_graph.deadline.time
        #         - (
        #             task.current_placement.placement_time.time
        #             + task.current_placement.execution_strategy.runtime.time
        #         )
        #         for task in task_graph.get_sink_tasks()
        #     ]
        # )
        # should_reschedule = (
        #     completion_slack >= task_graph.critical_path_runtime.time * allowed_slack
        # )

        # Strategy 3: Choose the TaskGraphs to be scheduled randomly with the given
        # sample size.
        task_graphs_for_scheduling.extend(
            np.random.choice(
                reschedulable_task_graphs,
                self._selectively_choose_task_graphs_sample_size,
                replace=False,
            )
        )

        # Strategy 4: Flip a random biased coin with a probability proportional to the
        # remaining slack in the task graph. The higher the slack, the higher the
        # probability of rescheduling.
        # slacks = []
        # total_slack = 0
        # for task_graph_name in reschedulable_task_graphs:
        #     schedulable_task_graph = workload.get_task_graph(task_graph_name)
        #     if schedulable_task_graph is None:
        #         raise ValueError(
        #             f"Could not find the TaskGraph {task_graph_name} in the workload."
        #         )
        #     slack = max(0, (schedulable_task_graph.deadline - current_time).time)
        #     slacks.append(slack)
        #     total_slack += slack

        # task_graphs_for_scheduling.extend(
        #     np.random.choice(
        #         reschedulable_task_graphs,
        #         self._selectively_choose_task_graphs_sample_size,
        #         replace=False,
        #         p=[slack / total_slack for slack in slacks],
        #     )
        # )

        # coin_probability = 0.5
        # total_slack = sum(slacks)
        # if total_slack > 0:
        #     coin_probability = (task_graph.deadline - current_time).time / total_slack
        # self._logger.debug(
        #     "[%s] The slack for TaskGraph %s was %s. The total slack was %s. "
        #     "The coin probability was %s.",
        #     current_time.time,
        #     task_graph.name,
        #     task_graph.deadline - current_time,
        #     total_slack,
        #     coin_probability,
        # )
        # should_reschedule = random.random() < coin_probability

        return task_graphs_for_scheduling

    def construct_task_graph_strl(
        self,
        current_time: EventTime,
        task_graph: TaskGraph,
        partitions: Partitions,
        placement_times_and_rewards: List[Tuple[EventTime, float]],
        tasks_to_be_scheduled: Optional[List[Task]] = None,
        task_strls: Optional[Mapping[str, tetrisched.strl.Expression]] = None,
        previously_placed: Optional[bool] = False,
        use_indicator_utility: Optional[bool] = False,
        scale_reward_previously_placed: float = 1.0,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph.

        Args:
            current_time (`EventTime`): The time at which the scheduling is occurring.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.
            partitions (`Partitions`): The partitions that are available for scheduling.
            tasks_to_be_scheduled (`Optional[List[Task]]`): The list of Tasks that are
                to be scheduled. If `None`, then all the Tasks in the TaskGraph are
                considered. Defaults to `None`.
            previously_placed (`Optional[bool]`): Whether the TaskGraph has been
                previously placed. Defaults to `False`.
            use_indicator_utility (`Optional[bool]`): Whether to use the indicator of
                the placement of the TaskGraph as the utility. If False, the utility
                of the individual Task placements is utilized instead.
            scale_reward_previously_placed (`float`): The factor by which to scale the
                reward of the TaskGraph if it has been previously placed. Defaults to
                1.0.
        """
        # Maintain a cache to be used across the construction of the TaskGraph to make
        # it DAG-aware, if not provided.
        if task_strls is None:
            task_strls = {}

        # Construct the STRL expression for all the roots of the TaskGraph.
        root_task_strls = {}
        strl_construction_success = True
        for root in task_graph.get_source_tasks():
            self._logger.debug(
                f"[{current_time.time}] Constructing the STRL for root "
                f"{root.unique_name} while creating the STRL for "
                f"TaskGraph {task_graph.name}."
            )
            root_task_strl = self._construct_task_graph_strl(
                current_time,
                root,
                task_graph,
                partitions,
                task_strls,
                placement_times_and_rewards,
                tasks_to_be_scheduled,
            )
            if root_task_strl is None:
                if tasks_to_be_scheduled is None or root in tasks_to_be_scheduled:
                    # If this is a root that we need to schedule, then we should fail
                    # the construction of the TaskGraph.
                    strl_construction_success = False
                    break
            else:
                root_task_strls[root_task_strl.id] = root_task_strl

        task_graph_strl = None
        if len(root_task_strls) == 0 or not strl_construction_success:
            # No roots, possibly empty TaskGraph, return None.
            return None
        elif len(root_task_strls) == 1:
            # Single root, reduce constraints and just bubble this up.
            task_graph_strl = next(iter(root_task_strls.values()))
        else:
            # Construct a MinExpression to order the roots of the TaskGraph.
            self._logger.debug(
                f"[{current_time.time}] Collecting {len(root_task_strls)} STRLs "
                f"for {task_graph.name} into a MinExpression "
                f"{task_graph.name}_min_expression."
            )
            min_expression_task_graph = tetrisched.strl.MinExpression(
                f"{task_graph.name}_min_expression"
            )
            for root_task_strl in root_task_strls.values():
                min_expression_task_graph.addChild(root_task_strl)
            task_graph_strl = min_expression_task_graph

        # Find which utility to assign to the TaskGraph and how to scale it.
        # The utility needs to be scaled if we request previously placed TaskGraph
        # utilities to be scaled, or if we are using the indicator from the topmost
        # TaskGraph expression to scale the utility.
        should_scale = (
            scale_reward_previously_placed > 1.0 and previously_placed
        ) or use_indicator_utility

        if should_scale:
            self._logger.debug(
                "[%s] Scaling the %s of %s by %s.",
                current_time.to(EventTime.Unit.US).time,
                "indicator" if use_indicator_utility else "utility",
                task_graph.name,
                scale_reward_previously_placed,
            )
            scale_expression = tetrisched.strl.ScaleExpression(
                f"{task_graph.name}_scale",
                scale_reward_previously_placed if previously_placed else 1,
                use_indicator_utility,
            )
            scale_expression.addChild(task_graph_strl)
            return scale_expression
        else:
            return task_graph_strl
