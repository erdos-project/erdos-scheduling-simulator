import multiprocessing
import sys
import time
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from uuid import UUID

import absl  # noqa: F401
import gurobipy as gp
import numpy as np
from gurobipy import GRB, and_

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
)


class PrimalDataPoint:
    """A `PrimalDataPoint` is used to represent a single data point in the search space
    for the solution of the primal problem described below.

    This class is intended to be used to store the progress of the search at particular
    points in time, and plot a primal integral gap figure later.
    """

    def __init__(
        self,
        start_time: float,
        datapoint_time: float,
        objective_value: float,
        objective_bound: float,
    ):
        self._start_time = start_time
        self._datapoint_time = datapoint_time
        self._objective_value = objective_value
        self._objective_bound = objective_bound

    @property
    def elapsed(self) -> EventTime:
        return EventTime(
            int((self._datapoint_time - self._start_time) * 1e6),
            EventTime.Unit.US,
        )

    def __str__(self) -> str:
        return (
            "PrimalDataPoint(elapsed={}, objective_value={}, "
            "objective_bound={})".format(
                self.elapsed,
                self._objective_value,
                self._objective_bound,
            )
        )


class TaskOptimizerVariables:
    """TaskOptimizerVariables is used to represent the optimizer variables for
    every particular task to be scheduled by the Scheduler.

    The initialization of this instance sets up the basic task-only constraints
    required by the problem.

    Args:
        current_time (`EventTime`): The time at which the scheduler was invoked.
            This is used to set a lower bound on the placement time of the tasks.
        plan_ahead (`EventTime`): The time frame to consider for scheduling decisions.
            If -1, uses whichever is greater: the greatest deadline or the greatest,
            remaining time across all tasks.
        time_discretization (`EventTime`): Assigns time to buckets to reduce the
            number of constraints. Defaults to 1 us.
        task (`Task`): The Task instance for which the variables are generated.
        workers (`Mapping[int, Worker]`): A mapping of the unique index of the
            Worker to its instance.
        optimizer (`gp.Model`): The instance of the Gurobi model to which the
            variables and constraints must be added.
        enforce_deadlines (`bool`): If `True`, the scheduler tries to enforce
            deadline constraints on the tasks.
    """

    def __init__(
        self,
        optimizer: gp.Model,
        task: Task,
        workers: Mapping[int, Worker],
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        enforce_deadlines: bool = True,
        retract_schedules: bool = False,
    ):
        self._task = task
        self._previously_placed = False
        self._start_time = None
        self._is_placed_variable = None

        # Placement characteristics
        # Indicator variables which signify the task's start time.
        time_range = range(
            current_time.to(EventTime.Unit.US).time,
            current_time.to(EventTime.Unit.US).time
            + plan_ahead.to(EventTime.Unit.US).time
            + 1,
            time_discretization.to(EventTime.Unit.US).time,
        )
        self._space_time_strategy_matrix = {
            (worker_id, t, strategy): 0
            for worker_id in workers.keys()
            for t in time_range
            for strategy in self._task.available_execution_strategies
        }
        self._placement_rewards = dict(
            zip(
                time_range,
                np.interp(time_range, (min(time_range), max(time_range)), (2, 1)),
            )
        )

        # Timing characteristics.
        if task.state == TaskState.RUNNING:
            # The task is already running, set the current time as the placed time and
            # the current worker as the placed worker. Set all of the remaining
            # combinations to 0.
            self._previously_placed = True
            placed_key = (
                self.__get_worker_index_from_previous_placement(task, workers),
                current_time.to(EventTime.Unit.US).time,
                task.current_placement.execution_strategy,
            )
            self._space_time_strategy_matrix[placed_key] = 1
            self._start_time = current_time.to(EventTime.Unit.US).time

            self._is_placed_variable = 1
        else:
            # Initialize all the possible placement opportunities for this task into the
            # space-time matrix. The worker has to be able to accomodate the task, and
            # the timing constraints as requested by the Simulator have to be met.
            schedulable_workers_to_strategies: Mapping[int, ExecutionStrategies] = {}
            for worker_id, worker in workers.items():
                cleared_worker = deepcopy(worker)
                compatible_strategies = cleared_worker.get_compatible_strategies(
                    self.task.available_execution_strategies
                )
                if len(compatible_strategies) > 0:
                    schedulable_workers_to_strategies[worker_id] = compatible_strategies

            for (
                worker_id,
                start_time,
                strategy,
            ) in self._space_time_strategy_matrix.keys():
                if (
                    worker_id not in schedulable_workers_to_strategies
                    or strategy not in schedulable_workers_to_strategies[worker_id]
                ):
                    # The Worker cannot accomodate this task or this strategy, and so
                    # the task should not be placed on this Worker.
                    self._space_time_strategy_matrix[
                        (worker_id, start_time, strategy)
                    ] = 0
                    continue

                if start_time < task.release_time.to(EventTime.Unit.US).time:
                    # The time is before the task gets released, the task cannot be
                    # placed here.
                    self._space_time_strategy_matrix[
                        (worker_id, start_time, strategy)
                    ] = 0
                elif (
                    enforce_deadlines
                    and start_time + strategy.runtime.to(EventTime.Unit.US).time
                    > task.deadline.to(EventTime.Unit.US).time
                ):
                    # The scheduler is asked to only place a task if the deadline can
                    # be met, and scheduling at this particular start_time leads to a
                    # deadline violation, so the task cannot be placed here.
                    self._space_time_strategy_matrix[
                        (worker_id, start_time, strategy)
                    ] = 0
                else:
                    # The placement needs to be decided by the optimizer.
                    self._space_time_strategy_matrix[
                        (worker_id, start_time, strategy)
                    ] = optimizer.addVar(
                        vtype=GRB.BINARY,
                        name=(
                            f"{task.unique_name}_placed_at_Worker"
                            f"_{worker_id}_on_Time_{start_time}"
                            f"_with_strategy_{strategy.id}"
                        ),
                    )

            # For each of the time slots, we generate a variable that defines whether
            # the task was placed at that time slot or not.
            self._time_slot_variables = {}
            for time_index in time_range:
                binary_variables_for_time_index = [
                    binary_variable
                    for (
                        _,
                        t,
                        _,
                    ), binary_variable in self._space_time_strategy_matrix.items()
                    if t == time_index
                ]
                task_placed_at_time_index = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{task.unique_name}_placed_at_time_{time_index}",
                )
                optimizer.addConstr(
                    task_placed_at_time_index
                    == gp.quicksum(binary_variables_for_time_index),
                    name=f"{task.unique_name}_placed_at_time_{time_index}_constraint",
                )
                task_not_placed_at_time_index = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{task.unique_name}_not_placed_at_time_{time_index}",
                )
                optimizer.addConstr(
                    task_not_placed_at_time_index == 1 - task_placed_at_time_index,
                    name=f"{task.unique_name}_not_placed_at"
                    f"_time_{time_index}_constraint",
                )
                self._time_slot_variables[time_index] = (
                    task_placed_at_time_index,
                    task_not_placed_at_time_index,
                )

            # We now pairwise check if the task was placed in a time slot B and is not
            # placed in an immediately preceding time slot A. If so, we set the start
            # time to be the time slot B. Without the ability to piece-wise schedule
            # the task (by preempting it), this should only be true once.
            self._start_time = optimizer.addVar(
                vtype=GRB.INTEGER,
                name=f"{task.unique_name}_start_time",
            )
            for time_index_a, time_index_b in zip(time_range, time_range[1:]):
                _, not_placed_at_a = self._time_slot_variables[time_index_a]
                placed_at_b, _ = self._time_slot_variables[time_index_b]
                phase_shift_at_index_b = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{task.unique_name}_phase_shift_at_time_{time_index_b}",
                )
                optimizer.addConstr(
                    phase_shift_at_index_b == and_(not_placed_at_a, placed_at_b),
                    name=f"{task.unique_name}_phase_shift_at_"
                    f"time_{time_index_b}_constraint",
                )
                optimizer.addGenConstrIndicator(
                    phase_shift_at_index_b,
                    1,
                    self._start_time,
                    GRB.EQUAL,
                    time_index_b,
                    name=f"{task.unique_name}_start_at_{time_index_b}_indicator",
                )
            # If no phase shifts are detected, the task may start at the first slot.
            optimizer.addGenConstrIndicator(
                self._time_slot_variables[time_range[0]][0],
                1,
                self._start_time,
                GRB.EQUAL,
                time_range[0],
                name=f"{task.unique_name}_start_at_{time_range[0]}_indicator",
            )

            if task.state == TaskState.SCHEDULED:
                # Seed the previously computed placement to the optimizer.
                placed_key = (
                    self.__get_worker_index_from_previous_placement(task, workers),
                    task.expected_start_time.to(EventTime.Unit.US).time,
                    task.current_placement.execution_strategy,
                )
                for (
                    space_time_index,
                    binary_variable,
                ) in self._space_time_strategy_matrix.items():
                    if not isinstance(binary_variable, gp.Var):
                        continue
                    if space_time_index == placed_key:
                        binary_variable.Start = 1
                    else:
                        binary_variable.Start = 0
                self._start_time.Start = task.expected_start_time.to(
                    EventTime.Unit.US
                ).time

            if task.state == TaskState.SCHEDULED and not retract_schedules:
                # If the task was previously scheduled, and we do not allow retractions,
                # we allow the start time to be fungible, but the task must be placed.
                optimizer.addConstr(
                    gp.quicksum(self._space_time_strategy_matrix.values()) == 1,
                    name=f"{task.unique_name}_previously_scheduled_"
                    f"required_worker_placement",
                )
                self._is_placed_variable = 1
            else:
                # If either the task was not previously placed, or we are allowing
                # retractions, then the task can be placed or left unplaced.
                optimizer.addConstr(
                    gp.quicksum(self._space_time_strategy_matrix.values()) <= 1,
                    name=f"{task.unique_name}_consistent_worker_placement",
                )
                self._is_placed_variable = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{task.unique_name}_is_placed",
                )
                optimizer.addConstr(
                    self._is_placed_variable
                    == gp.quicksum(self._space_time_strategy_matrix.values()),
                    name=f"{task.unique_name}_is_placed_constraint",
                )

    def __get_worker_index_from_previous_placement(
        self, task: Task, workers: Mapping[int, Worker]
    ) -> int:
        """Maps the ID of the Worker that the Task was previously placed to the index
        that it was assigned for this invocation of the Scheduler.

        Args:
            task (`Task`): The Task for which the previous placed Worker is to be
                retrieved.
            workers (`Mapping[int, Worker]`): The current mapping of indices to the
                Workers.

        Returns:
            The index of the Worker in this instance of the Scheduler, if found.
            Otherwise, a ValueError is raised with the appropriate information.
        """
        if task.current_placement is None or task.current_placement.worker_id is None:
            raise ValueError(
                f"Task {task.unique_name} in state {task.state} does not have a "
                f"cached prior Placement or the Worker ID is empty."
            )
        worker_index = None
        for worker_id, worker in workers.items():
            if worker.id == task.current_placement.worker_id:
                worker_index = worker_id
                break
        if worker_index is None:
            raise ValueError(
                f"Task {task.unique_name} in state {task.state} was previously placed "
                f"on {task.current_placement.worker_id}, which was no longer found in "
                f"the current set of available Workers."
            )
        return worker_index

    def get_placements(
        self,
        worker_index_to_worker: Mapping[int, Worker],
        worker_id_to_worker_pool: Mapping[UUID, UUID],
    ) -> Sequence[Placement]:
        """Retrieves the details of the solution from the optimizer and constructs the
        `Placement` objects for the Scheduler to return to the Simulator.

        Args:
            worker_index_to_worker (`Mapping[int, Worker]`): A mapping from the index
                that the Worker was assigned for this scheduling run to a reference to
                the `Worker` itself.
            worker_id_to_worker_pool (`Mapping[UUID, UUID]`): A mapping from the ID of
                the `Worker` to the ID of the `WorkerPool` which it is a part of.

        Returns:
            A sequence of `Placement` objects depicting the time when the Task(s) are
            to be started, and the Worker where the Task(s) are to be executed.
        """
        if self.previously_placed:
            # If the task was previously placed, then we don't have to return a
            # `Placement` object to the Simulator.
            return []

        for (
            worker_id,
            start_time,
            strategy,
        ), variable in self._space_time_strategy_matrix.items():
            if type(variable) != int and variable.X == 1:
                placement_worker = worker_index_to_worker[worker_id]
                placement_worker_pool_id = worker_id_to_worker_pool[placement_worker.id]
                return [
                    Placement.create_task_placement(
                        task=self.task,
                        placement_time=EventTime(start_time, unit=EventTime.Unit.US),
                        worker_pool_id=placement_worker_pool_id,
                        worker_id=placement_worker.id,
                        execution_strategy=strategy,
                    )
                ]
        return [
            Placement.create_task_placement(
                task=self.task,
                placement_time=None,
                worker_pool_id=None,
                worker_id=None,
                execution_strategy=None,
            )
        ]

    def get_partition_variable(
        self, time: int, worker_index: int
    ) -> Mapping[ExecutionStrategy, Sequence[Union[gp.Var, int]]]:
        """Get the set of variables that can potentially affect the resource
        utilization of the worker with the given index `worker_index` at the time
        `time`.

        Args:
            time (`int`): The time at which the effect is to be determined.
            worker_index (`int`): The index of the Worker to determine the effect for
                in the schedule.

        Returns:
            A possibly empty sequence of either cplex variables or integers specifying
            all the possible placement options of the given task that may affect the
            usage of resources on the given Worker at the given time, mapped to the
            strategy with which they are being associated..
        """
        partition_variables = defaultdict(list)
        for (
            worker_id,
            start_time,
            strategy,
        ), variable in self._space_time_strategy_matrix.items():
            if (
                worker_id == worker_index
                and start_time <= time
                and start_time + strategy.runtime.to(EventTime.Unit.US).time > time
                and (type(variable) == gp.Var or variable == 1)
            ):
                partition_variables[strategy].append(variable)
        return partition_variables

    @property
    def start_time(self) -> Union[int, gp.Var]:
        """Returns a Gurobi variable representing the start time for this Task."""
        return self._start_time

    @property
    def task(self) -> Task:
        """Returns the Task that this instance of TaskOptimizerVariables represents."""
        return self._task

    @property
    def name(self) -> str:
        """Returns the name of the Task that this instance represents."""
        return self._task.unique_name

    @property
    def previously_placed(self) -> bool:
        """Returns a boolean variable indicating if the Task that this instance
        represents was previously placed (`True`), or being considered for scheduling
        during this execution (`False`)."""
        return self._previously_placed

    @property
    def space_time_matrix(
        self,
    ) -> Mapping[Tuple[int, int, ExecutionStrategy], Union[int, gp.Var]]:
        """Returns a mapping from the (Worker Index, Time, ExecutionStrategy) to the
        binary variable specifying if the task was placed at that time or not.
        """
        return self._space_time_strategy_matrix

    @property
    def is_placed(self) -> Optional[Union[int, gp.Var]]:
        """Returns the binary variable that specifies if the task was placed or not."""
        return self._is_placed_variable


class TetriSchedGurobiScheduler(BaseScheduler):
    """Implements an TetriSched formulation of the scheduling problem for the Simulator
    using Gurobi.

    Args:
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
        goal (`str`): The goal to use as the optimization objective.
        time_limit (`EventTime`): The time to keep searching for new solutions without
            any changes to either the incumbent or the best bound.
        time_discretization (`EventTime`): Assigns time to buckets to reduce the
            number of constraints. Defaults to 1 us.
        plan_ahead (`EventTime`): The time frame to consider for scheduling decisions.
            If -1, uses the greatest deadline.
        log_to_file (`bool`): If `True`, the scheduler writes the Gurobi search
            log to files with the format "gurobi_{sim_time}.log".
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime.invalid(),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = True,
        release_taskgraphs: bool = False,
        goal: str = "max_goodput",
        batching: bool = False,
        time_limit: EventTime = EventTime(time=20, unit=EventTime.Unit.S),
        time_discretization: EventTime = EventTime(time=1, unit=EventTime.Unit.US),
        plan_ahead: EventTime = EventTime.invalid(),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        if preemptive or batching:
            raise ValueError(
                "TetriSchedGurobiScheduler does not support preemptive or "
                "batching-aware scheduling."
            )
        super(TetriSchedGurobiScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        self._goal = goal
        self._batching = batching
        self._gap_time_limit = time_limit
        self._time_discretization = time_discretization
        self._plan_ahead = plan_ahead
        self._log_to_file = log_to_file
        self._log_times = set(map(int, _flags.scheduler_log_times)) if _flags else set()

    def _initialize_optimizer(self, current_time: EventTime) -> gp.Model:
        """Initializes the Optimizer and sets the required parameters.

        Args:
            current_time (`EventTime`): The time at which the model was supposed
                to be invoked.

        Returns:
            An optimizer of type `gp.Model` to whom the variables and constraints
            can be added.
        """
        optimizer = gp.Model("TetriSched")

        # Don't log the output to the console, instead log it to a file.
        optimizer.Params.LogToConsole = 0
        if (
            self._log_to_file
            or current_time.to(EventTime.Unit.US).time in self._log_times
        ):
            optimizer.Params.LogFile = (
                f"./tetrisched_gurobi_{current_time.to(EventTime.Unit.US).time}.log"
            )

        # If the goal is goodput, set the MIPGap to 0.1.
        if self._goal == "max_goodput":
            optimizer.Params.MIPGap = 0.1

        # Always decide between INFINITE or UNBOUNDED.
        optimizer.Params.DualReductions = 0

        # Ask Gurobi to focus on finding solutions rather than proving optimality.
        optimizer.Params.MIPFocus = 1

        # Ask Gurobi to aggressively cut the search space for proving optimality.
        optimizer.Params.Cuts = 3

        # Set the number of threads for this machine.
        optimizer.Params.Threads = multiprocessing.cpu_count()

        return optimizer

    def schedule(
        self,
        sim_time: EventTime,
        workload: Workload,
        worker_pools: WorkerPools,
    ) -> Placements:
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            release_taskgraphs=self.release_taskgraphs,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
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

        # Create a virtual WorkerPool set to try scheduling decisions on.
        schedulable_worker_pools = copy(worker_pools)

        # Construct a mapping of the index of a Worker to the Worker itself, and
        # a mapping from the Worker to the WorkerPool to which it belongs.
        worker_index = 1
        workers = {}
        worker_to_worker_pool = {}
        for worker_pool in schedulable_worker_pools.worker_pools:
            for worker in worker_pool.workers:
                workers[worker_index] = worker
                worker_index += 1
                worker_to_worker_pool[worker.id] = worker_pool.id

        self._logger.debug(
            "[{}] The scheduler received {} tasks for scheduling across {} workers. "
            "These tasks were: {}.".format(
                sim_time.time,
                len(tasks_to_be_scheduled),
                len(workers),
                [
                    f"{task.unique_name} ({task.deadline})"
                    for task in tasks_to_be_scheduled
                ],
            )
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler is also considering the following "
            f"{len(previously_placed_tasks)} for their effects on the current "
            f"placements: {[task.unique_name for task in previously_placed_tasks]}."
        )

        # Construct the model and the variables for each of the tasks.
        scheduler_start_time = time.time()
        placements = []
        if len(tasks_to_be_scheduled) != 0 and not all(
            [t.state == TaskState.SCHEDULED for t in tasks_to_be_scheduled]
        ):
            optimizer = self._initialize_optimizer(sim_time)
            tasks_to_variables = self._add_variables(
                sim_time=sim_time,
                optimizer=optimizer,
                tasks_to_be_scheduled=tasks_to_be_scheduled + previously_placed_tasks,
                workers=workers,
            )

            # Add the constraints to ensure that dependecy constraints are met and
            # resources are not oversubscribed.
            self._add_task_dependency_constraints(
                sim_time, optimizer, tasks_to_variables, workload, workers
            )
            self._add_resource_constraints(
                sim_time, optimizer, tasks_to_variables, workers
            )

            # Add the objectives and optimize the model.
            self._add_objective(optimizer, tasks_to_variables, workload)

            if self._log_to_file or (
                sim_time.to(EventTime.Unit.US).time in self._log_times
            ):
                optimizer.write(
                    f"./tetrisched_gurobi_{sim_time.to(EventTime.Unit.US).time}.lp"
                )
            optimizer_start_time = time.time()
            optimizer.optimize(
                callback=lambda optimizer, where: self._termination_check_callback(
                    sim_time, optimizer, where
                )
            )
            optimizer_end_time = time.time()

            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned the "
                f"status {optimizer.status} in "
                f"{int((optimizer_end_time - optimizer_start_time) * 1e6)}µs."
            )

            # Collect the placement results.
            if optimizer.Status == GRB.OPTIMAL or (
                optimizer.Status == GRB.INTERRUPTED and optimizer._solution_found
            ):
                for primal_data_point in sorted(
                    optimizer._primal_data_points,
                    key=lambda dp: dp._datapoint_time,
                ):
                    self._logger.debug(
                        "[{}] The solver found a solution: {}.".format(
                            sim_time.to(EventTime.Unit.US).time,
                            str(primal_data_point),
                        )
                    )

                self._logger.debug(
                    f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned "
                    f"the objective value {optimizer.objVal}."
                )

                # Write the solution to the SOL file, if requested.
                if self._log_to_file or (
                    sim_time.to(EventTime.Unit.US).time in self._log_times
                ):
                    optimizer.write(
                        f"./tetrisched_gurobi_{sim_time.to(EventTime.Unit.US).time}.sol"
                    )

                for task_variables in tasks_to_variables.values():
                    placements_for_task: Sequence[
                        Placement
                    ] = task_variables.get_placements(
                        worker_index_to_worker=workers,
                        worker_id_to_worker_pool=worker_to_worker_pool,
                    )
                    for placement in placements_for_task:
                        if placement.is_placed():
                            self._logger.debug(
                                "[%s] Placed %s (with deadline %s and "
                                "remaining time %s) on WorkerPool(%s) to be "
                                "started at %s and executed with strategy %s (%s).",
                                sim_time.to(EventTime.Unit.US).time,
                                placement.task.unique_name,
                                placement.task.deadline,
                                placement.execution_strategy.runtime,
                                placement.worker_pool_id,
                                placement.placement_time,
                                placement.execution_strategy,
                                placement.execution_strategy.id,
                            )
                        else:
                            self._logger.debug(
                                "[%s] Failed to find a valid Placement for %s.",
                                sim_time.to(EventTime.Unit.US).time,
                                placement.task.unique_name,
                            )
                        placements.append(placement)
            else:
                # No feasible solution was found, cancel the tasks that were required
                # to be scheduled, and assume the prior placement for all the other
                # tasks.
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
                self._logger.warning(f"[{sim_time.time}] Failed to place any task.")
        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6),
            EventTime.Unit.US,
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
            runtime=runtime,
            true_runtime=scheduler_runtime,
            placements=placements,
        )

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        tasks_to_be_scheduled: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
        """Generates the variables for the optimization problem.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            optimizer (`gp.Model`): The instance of the Gurobi model to which the
                variables and constraints must be added.
            tasks_to_be_scheduled (`Sequence[Task]`): The tasks to be scheduled
                in this scheduling run.
            workers (`Mapping[int, Worker]`): The collection of Workers to schedule
                the tasks on.

        Returns:
            A mapping from the unique name of the Task to the variables representing
            its optimization objective.
        """
        tasks_to_variables: Mapping[str, TaskOptimizerVariables] = {}
        plan_ahead = self._plan_ahead
        if plan_ahead == EventTime.invalid():
            for task in tasks_to_be_scheduled:
                if task.deadline > plan_ahead:
                    plan_ahead = task.deadline

        for task in tasks_to_be_scheduled:
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                optimizer=optimizer,
                task=task,
                workers=workers,
                current_time=sim_time,
                plan_ahead=plan_ahead,
                time_discretization=self._time_discretization,
                enforce_deadlines=self.enforce_deadlines,
                retract_schedules=self.retract_schedules,
            )
        return tasks_to_variables

    def _add_task_dependency_constraints(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
        workers: Mapping[int, Worker],
    ) -> None:
        """Generates the variables and constraints to ensure that the dependencies
        due to a TaskGraph are respected.

        Args:
            sim_time (`EventTime`): The time at which the scheduler is being invoked.
            optimizer (`gp.Model`): The instance of the Gurobi model to which the
                variables and constraints must be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): A mapping
                from the name of the Task to its corresponding variables inside the
                optimizer.
            workload (`Workload`): The workload with which the scheduler was invoked.
            workers (`Mapping[int, Worker]`): A mapping of the unique index of the
                Worker to its instance.
        """
        for task_name, task_variable in tasks_to_variables.items():
            if task_variable.previously_placed:
                # The task was previously placed, we should not add any constraints
                # for this Task.
                continue

            # A mapping of the parent variables of the Task to the number of parents
            # that are present in those set of variables.
            parent_variables: Mapping[TaskOptimizerVariables, int] = {}
            # If the task is a `Task`, we need to ensure the following:
            # 1. The Task is only placed if all of its parents are placed.
            # 2. The Task is started after the last parent has finished.
            task_graph = workload.get_task_graph(task_variable.task.task_graph)
            parent_tasks = set(task_graph.get_parents(task_variable.task))
            for variable in tasks_to_variables.values():
                # Get the number of parent tasks in this variable.
                if variable.task in parent_tasks:
                    num_parents_in_variable = 1
                else:
                    num_parents_in_variable = 0

                if num_parents_in_variable > 0:
                    parent_variables[variable] = num_parents_in_variable

            # Ensure that the task is only placed if all of its parents are placed,
            # and that it is started after the last parent has finished.
            if len(parent_variables) > 0:
                all_parents_placed = optimizer.addVar(
                    vtype=GRB.BINARY, name=f"{task_name}_all_parents_placed"
                )
                for (
                    parent_variable,
                    num_parents_in_variable,
                ) in parent_variables.items():
                    if parent_variable.previously_placed:
                        parent_remaining_time = (
                            parent_variable.task.remaining_time.to(
                                EventTime.Unit.US
                            ).time
                            + 1
                        )
                        optimizer.addConstr(
                            task_variable.start_time
                            >= parent_variable.start_time + parent_remaining_time,
                            name=f"{task_name}_start_after_running_task_"
                            f"{parent_variable.name}_remaining_time_"
                            f"{parent_remaining_time}",
                        )
                    else:
                        parent_strategies = (
                            parent_variable.task.available_execution_strategies
                        )
                        parent_strategy = parent_strategies.get_slowest_strategy()
                        optimizer.addConstr(
                            task_variable.start_time
                            >= parent_variable.start_time
                            + parent_strategy.runtime.to(EventTime.Unit.US).time
                            + 1,
                            name=f"{task_name}_start_after_{parent_variable.name}",
                        )

                # Construct an indicator variable that checks if all the parents were
                # placed on some worker or not.
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    0,
                    gp.quicksum(
                        parent_variable.is_placed
                        for parent_variable in parent_variables
                    ),
                    GRB.LESS_EQUAL,
                    len(parent_tasks) - 1,
                    name=f"{task_name}_parents_placed_False",
                )
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    1,
                    gp.quicksum(
                        parent_variable.is_placed
                        for parent_variable in parent_variables
                    ),
                    GRB.EQUAL,
                    len(parent_tasks),
                    name=f"{task_name}_parents_placed_True",
                )

                # If all of the parents were not placed, then we cannot place this task.
                # Otherwise, we allow the optimizer to choose when to place the task.
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    0,
                    task_variable.is_placed,
                    GRB.EQUAL,
                    0,
                    name=f"{task_name}_placement_False",
                )

    def _add_resource_constraints(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workers: Mapping[int, Worker],
    ) -> None:
        """Generates the variables and constraints to ensure that the resource
        availability inside the Workers is not violated.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            optimizer (`gp.Model`): The instance of the Gurobi model to which the
                variables and constraints must be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): A mapping
                from the name of the Task to its corresponding variables inside the
                optimizer.
            workers (`Mapping[int, Worker]`): A mapping of the unique index of the
                Worker to its instance.
        """
        plan_ahead = self._plan_ahead
        if plan_ahead == EventTime.invalid():
            for task_variables in tasks_to_variables.values():
                if task_variables.task.deadline > plan_ahead:
                    plan_ahead = task_variables.task.deadline

        for t in range(
            sim_time.to(EventTime.Unit.US).time,
            sim_time.to(EventTime.Unit.US).time
            + plan_ahead.to(EventTime.Unit.US).time
            + 1,
            self._time_discretization.to(EventTime.Unit.US).time,
        ):
            for worker_index, worker in workers.items():
                # Get all the placement variables that affect the resource utilization
                # on this worker at this particular time.
                overlap_variables: Mapping[ExecutionStrategy, Any] = defaultdict(list)
                for task_variable in tasks_to_variables.values():
                    partition_variables = task_variable.get_partition_variable(
                        time=t,
                        worker_index=worker_index,
                    )
                    for strategy, variables in partition_variables.items():
                        overlap_variables[strategy].extend(variables)

                # For each resource in the Worker, find the resource request for all
                # the tasks and multiply it by the partition variables to ensure that
                # resources are never oversubscribed.
                for (
                    resource,
                    quantity,
                ) in worker.resources.get_unique_resource_types().items():
                    resource_constraint_expression = gp.LinExpr()
                    for (
                        strategy,
                        strategy_overlap_variables,
                    ) in overlap_variables.items():
                        strategy_request_for_resource = (
                            strategy.resources.get_total_quantity(resource)
                        )
                        if strategy_request_for_resource == 0:
                            # If the strategy does not require this resource, we skip
                            # adding any of the terms into this resource constraint
                            # expression.
                            continue
                        for strategy_overlap_variable in strategy_overlap_variables:
                            resource_constraint_expression.add(
                                strategy_request_for_resource
                                * strategy_overlap_variable
                            )
                    if quantity == 0 or resource_constraint_expression.size() == 0:
                        # If either the Worker doesn't have enough space to accomodate
                        # this task, or no task wants to occupy this resource at the
                        # particular time, we skip the addition of this constraint.
                        continue
                    optimizer.addConstr(
                        resource_constraint_expression <= quantity,
                        name=f"{resource.name}_utilization_Worker"
                        f"_{worker_index}_at_Time_{t}",
                    )

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        """Generates the constraints for the optimization objective as specified by
        the `goal` parameter to this instantiation of the Scheduler.

        Args:
            optimizer (`cpx.Model`): The optimization model to which the constraints
                are to be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): The mapping
                from the task names to the `TaskOptimizerVariables` instance that
                contains its representation of the space-time matrix.
        """
        if self._goal == "max_goodput":
            # Define reward variables for each of the tasks, that is a sum of their
            # space-time matrices. Maximizing the sum of these rewards is the goal of
            # the scheduler.
            task_reward_variables = []
            for task_variable in tasks_to_variables.values():
                task_reward_variables.extend(
                    [
                        task_variable._placement_rewards[t] * value
                        for (
                            _,
                            t,
                            _,
                        ), value in task_variable.space_time_matrix.items()
                    ]
                )
            optimizer.setObjective(
                gp.quicksum(task_reward_variables),
                sense=GRB.MAXIMIZE,
            )
        else:
            raise RuntimeError(
                f"The goal {self._goal} is not supported yet by "
                f"TetriSchedGurobiScheduler."
            )

        # Update the time at which the gap was updated and the gap itself.
        optimizer._solution_found = False
        optimizer._current_gap = float("inf")
        optimizer._last_gap_update = time.time()
        optimizer._solver_start_time = time.time()
        optimizer._primal_data_points = []

    def _termination_check_callback(
        self, sim_time: EventTime, optimizer: gp.Model, where: GRB.Callback
    ) -> None:
        """The callback that is invoked by the Gurobi optimizer at the end of every
        iteration to check if the search has to be terminated.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            optimizer (`gp.Model`): The model that is being optimized.
            where (`GRB.Callback`): The site where the callback was invoked.
        """
        if where == GRB.Callback.MIPSOL:
            # A new MIP solution has been found. Log the time at which it was found.
            new_solution_objective = optimizer.cbGet(GRB.Callback.MIPSOL_OBJ)
            best_solution_objective_bound = optimizer.cbGet(GRB.Callback.MIPSOL_OBJBND)
            current_time = time.time()
            primal_data_point = PrimalDataPoint(
                optimizer._solver_start_time,
                current_time,
                new_solution_objective,
                best_solution_objective_bound,
            )
            optimizer._primal_data_points.append(primal_data_point)

        if where == GRB.Callback.MIPNODE:
            # Retrieve the current bound and the solution from the model, and use it
            # to compute the Gap between the solution according to the formula.
            best_objective_bound = optimizer.cbGet(GRB.Callback.MIPNODE_OBJBND)
            incumbent_objective = optimizer.cbGet(GRB.Callback.MIPNODE_OBJBST)
            if incumbent_objective < sys.float_info.epsilon:
                # If we have no solution, assume the gap is infinity.
                gap = float("inf")
            else:
                gap = best_objective_bound - incumbent_objective / incumbent_objective

            # If the gap changed, update the time at which it changed.
            if abs(gap - optimizer._current_gap) > sys.float_info.epsilon:
                self._logger.debug(
                    "[%s] The gap between the incumbent (%f) and the best bound "
                    "(%f) was updated from %f to %f.",
                    sim_time.to(EventTime.Unit.US).time,
                    incumbent_objective,
                    best_objective_bound,
                    optimizer._current_gap,
                    gap,
                )
                optimizer._solution_found = True
                optimizer._current_gap = gap
                optimizer._last_gap_update = time.time()

        # If the gap hasn't changed in the predefined time, terminate the model.
        solver_time = time.time()
        if self._gap_time_limit.is_invalid():
            # The solver was asked to run indefinitely. Do not terminate.
            return
        gap_time_limit_s = self._gap_time_limit.to(EventTime.Unit.S).time
        if (
            solver_time - optimizer._last_gap_update > gap_time_limit_s
        ) and optimizer._solution_found:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't "
                "changed in %f seconds, and there is a valid solution. "
                "Terminating.",
                sim_time.to(EventTime.Unit.US).time,
                solver_time - optimizer._last_gap_update,
            )
            optimizer.terminate()
        elif solver_time - optimizer._last_gap_update > gap_time_limit_s:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't "
                "changed in %f seconds, and no valid solution has yet been "
                "found. Terminating.",
                sim_time.to(EventTime.Unit.US).time,
                solver_time - optimizer._last_gap_update,
            )
            optimizer.terminate()
