from collections import defaultdict
import itertools
import multiprocessing
import sys
import time
from copy import copy, deepcopy
from itertools import combinations
from typing import Mapping, MutableSet, Optional, Sequence, Tuple, Union
from uuid import UUID

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
)
from workload.resources import Resources


class SpaceTimeMatrix:
    """A matrix of variables representing allocation in space and time.

    Maps worker index and event time to a binary indicator.
    If the indicator has a value of 1, the space-time slot is occupied.
    If the indicator has a value of 0, the space-time slot is not occupied.
    The indicator may also be a gurobi variable, in which case the optimizer will solve
    for its value.

    Args:
        workers (`Mapping[int, Worker]`): A mapping of the unique index of the
            Worker to its instance.
        start_time (`EventTime`): Lower bound of the matrix's time frame.
        end_time (`EventTime`): Upper bound of the matrix's time frame.
        time_discretization (`EventTime`): The length of the discrete slices of time
            considered used to build the matrix.
    """

    def __init__(
        self,
        workers: Mapping[int, Worker],
        start_time: EventTime,
        end_time: EventTime,
        time_discretization: EventTime,
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._time_discretization = time_discretization
        self._num_buckets = self._event_time_to_time_bucket(end_time)
        self._matrix = {
            worker_idx: [0 for _ in range(self._num_buckets)] for worker_idx in workers
        }

    @property
    def start_time(self) -> EventTime:
        return self._start_time

    @property
    def end_time(self) -> EventTime:
        return self._end_time

    @property
    def time_discretization(self) -> EventTime:
        return self._time_discretization

    @property
    def worker_indices(self) -> Sequence[int]:
        return list(self._matrix.keys())

    @property
    def time_slices(self) -> Sequence[EventTime]:
        return list(map(self._time_bucket_to_event_time, range(self._num_buckets)))

    def _event_time_to_time_bucket(self, event_time: EventTime) -> int:
        assert event_time >= self._start_time
        relative_time = event_time - self._start_time
        return (
            relative_time.to(EventTime.Unit.US).time
            // self._time_discretization.to(EventTime.Unit.US).time
        )

    def _time_bucket_to_event_time(self, bucket: int) -> EventTime:
        event_time = self._start_time
        for _ in range(bucket):
            event_time += self._time_discretization
        return event_time

    def _parse_key(
        self,
        key: Tuple[
            int | Sequence[int] | slice, EventTime | Sequence[EventTime] | slice
        ],
    ) -> Tuple[Sequence[int], Sequence[EventTime]]:
        worker_idxs, event_times = key
        # Parse worker indices.
        if isinstance(worker_idxs, int):
            worker_idxs = [worker_idxs]
        elif isinstance(worker_idxs, Sequence):
            worker_idxs = list(worker_idxs)
        elif isinstance(worker_idxs, slice):
            if worker_idxs.step is not None:
                raise ValueError("Stride not supported for worker indexes.")
            start = (
                min(self._matrix.keys())
                if worker_idxs.start is None
                else worker_idxs.start
            )
            stop = (
                max(self._matrix.keys()) + 1
                if worker_idxs.stop is None
                else worker_idxs.stop
            )
            worker_idxs = list(
                sorted(filter(lambda x: x >= start and x < stop, self._matrix.keys()))
            )
        else:
            raise ValueError(f"Worker indices cannot be of type {type(worker_idxs)}")
        # Parse event times.
        if isinstance(event_times, EventTime):
            event_times = [event_times]
        elif isinstance(event_times, Sequence):
            event_times = list(event_times)
        elif isinstance(event_times, slice):
            start = self.start_time if event_times.start is None else event_times.start
            stop = self.end_time if event_times.stop is None else event_times.stop
            step = (
                self.time_discretization
                if event_times.step is None
                else event_times.step
            )
            event_times = [start]
            while event_times[-1] + step < stop:
                event_times.append(event_times[-1] + step)
        else:
            raise ValueError(f"Event times cannot be of type {type(event_times)}")
        return worker_idxs, event_times

    def __getitem__(
        self,
        key: Tuple[
            int | Sequence[int] | slice, EventTime | Sequence[EventTime] | slice
        ],
    ) -> int | gp.Var | Sequence[int | gp.Var]:
        worker_idxs, event_times = self._parse_key(key)
        event_time_buckets = list(map(self._event_time_to_time_bucket, event_times))
        results = {
            worker_idx: [
                self._matrix[worker_idx][time_bucket]
                for time_bucket in event_time_buckets
            ]
            for worker_idx in worker_idxs
        }
        if isinstance(key[1], EventTime):
            results = {k: v[0] for k, v in results.items()}
        if isinstance(key[0], int):
            results = results[key[0]]
        return results

    def __setitem__(self, key, value: int | gp.Var):
        if not isinstance(value, (int, gp.Var)):
            raise ValueError(f"Assigning to {type(value)} is not supported")
        worker_idxs, event_times = self._parse_key(key)
        for worker_idx in worker_idxs:
            for time_bucket in map(self._event_time_to_time_bucket, event_times):
                self._matrix[worker_idx][time_bucket] = value


class TaskOptimizerVariables:
    """TaskOptimizerVariables is used to represent the optimizer variables for
    every particular task to be scheduled by the Scheduler.

    The initialization of this instance sets up the basic task-only constraints
    required by the problem.

    Args:
        current_time (`EventTime`): The time at which the scheduler was invoked.
            This is used to set a lower bound on the placement time of the tasks.
        plan_ahead (`EventTime`): The time frame to consider for scheduling decisions.
            If -1, uses the greatest deadline.
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
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        task: Task,
        workers: Mapping[int, Worker],
        optimizer: gp.Model,
        enforce_deadlines: bool = True,
        retract_schedules: bool = False,
    ):
        self._task = task
        self._enforce_deadlines = enforce_deadlines
        self._start_time = None

        # Placement characteristics
        # Indicator variables which signify the task's start time.
        self._start_time_indicators = {}
        # Matrices of indicator variables that signify the times at which the task was
        # placed on each worker. Contains 1 matrix per possible start time.
        self._matrices = {}

        # Set up individual variables to signify where and when task is placed.
        self._placed_on_worker = {}

        # Timing characteristics.
        if task.state == TaskState.RUNNING:
            # The task is already running, set the start time to the current
            # simulation time, since we use the remaining time to count the
            # time at which this Task will relinquish its resources.
            self._previously_placed = True
            self._is_placed = 1
            self._start_time = current_time.to(EventTime.Unit.US).time
            self._start_time_indicators[current_time] = 1

            # Find the Worker where the Task was previously placed and add
            # constraints to inform the scheduler of its placement.
            if (
                task.current_placement is None
                or task.current_placement.worker_id is None
            ):
                raise ValueError(
                    f"Task {task.unique_name} in state {task.state} does not have a "
                    f"cached prior Placement or the Worker ID is empty."
                )

            previously_placed_worker = task.current_placement.worker_id
            matrix = SpaceTimeMatrix(
                workers, current_time, plan_ahead, time_discretization
            )
            matrix[
                previously_placed_worker,
                current_time : current_time + task.remaining_time,
            ] = 1
            self._matrices[current_time] = matrix
            self._start_time_indicators[current_time] = 1
        else:
            # The task was scheduled and we are not allowing retractions, we allow the
            # start time to be fungible but the task must be placed.
            self._previously_placed = False
            self._initialize_variables_and_constraints(
                optimizer,
                workers,
                enforce_deadlines,
                current_time,
                time_discretization,
                plan_ahead,
            )

            if task.state == TaskState.SCHEDULED:
                # Set up variables to signify if the Task was placed on a Worker.
                previously_placed_worker = task.current_placement.worker_id
                for worker_id, worker in workers.items():
                    if worker.id == previously_placed_worker:
                        self._placed_on_worker[worker_id].Start = 1
                    else:
                        self._placed_on_worker[worker_id].Start = 0

            if task.state == TaskState.SCHEDULED and not retract_schedules:
                # Add constraint that the task must be placed.
                optimizer.addConstr(
                    self.is_placed == 1, name=f"{task.unique_name}_must_be_placed"
                )

                # Compute expected start time.
                expected_start_time_us = max(
                    0,
                    (task.expected_start_time - current_time)
                    .to(EventTime.Unit.US)
                    .time,
                )
                time_discretization_us = time_discretization.to(EventTime.Unit.US).time
                expected_start_time_us_discretized = (
                    expected_start_time_us // time_discretization_us
                )
                expected_start_time_discretized = EventTime(
                    expected_start_time_us_discretized, EventTime.Unit.US
                )
                # Seed expected start time.
                for (
                    start_time,
                    start_time_indicator,
                ) in self._start_time_indicators.items():
                    if start_time == expected_start_time_discretized:
                        start_time_indicator.Start = 1
                    else:
                        start_time_indicator.Start = 0
                # Seed partition variables.
                for start_time, matrix in self._matrices.items():
                    if start_time == expected_start_time_discretized:
                        matrix[previously_placed_worker, start_time].Start = 1
                    else:
                        matrix[previously_placed_worker, start_time].Start = 0

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
        """Returns a Boolean indicating whether the task denoted by this instance
        has been previously placed or not."""
        return self._previously_placed

    @property
    def enforce_deadlines(self) -> bool:
        return self._enforce_deadlines

    @property
    def space_time_matrices(self) -> Mapping[EventTime, SpaceTimeMatrix]:
        """Mapping from start time to `SpaceTimeMatrix`."""
        return self._matrices

    @property
    def start_time_indicators(self) -> Mapping[EventTime, int | gp.Var]:
        """Mapping from start time to indicator whether the start time is chosen."""
        return self._start_time_indicators

    @property
    def is_placed(self) -> int | gp.Var:
        """Check if the scheduler found a placement for the task."""
        return self._is_placed

    def placed_on_worker(self, worker_id: int) -> Optional[Union[int, gp.Var]]:
        """Check if the Task was placed on a particular Worker.

        Args:
            worker_id (`int`): The ID of the Worker to check.

        Returns:
            The method has the following return values:
                - `None`: If the `Worker` ID was not registered with the instance.
                - `int`: If the `Task` was forced to not be placed on this `Worker`.
                - `gp.Var`: A Gurobi variable representing the placement solution.
        """
        return self._placed_on_worker.get(worker_id)

    @property
    def placed_on_workers(self) -> Sequence[Union[int, gp.Var]]:
        """Retrieves the binary Gurobi variables representing the placement of the
        task on the collection of `Worker`s registered with the instance."""
        return self._placed_on_worker.values()

    def __str__(self) -> str:
        return f"TaskOptimizerVariables(name={self.name})"

    def __repr__(self) -> str:
        return str(self)

    def _initialize_variables_and_constraints(
        self,
        optimizer: gp.Model,
        workers: Mapping[int, Worker],
        enforce_deadlines: bool,
        current_time: EventTime,
        time_discretization: EventTime,
        plan_ahead: EventTime,
    ) -> None:
        # Add a constraint to ensure that if enforcing deadlines is required, then
        # we are setting the start time to something that meets the deadline.
        self._is_placed = optimizer.addVar(
            vtype=GRB.BINARY, name=f"{self.name}_is_placed"
        )
        if enforce_deadlines:
            end_time = self.task.deadline
        else:
            assert plan_ahead > self.task.remaining_time, (
                f"Cannot schedule task because plan-ahead {plan_ahead} is shorter than"
                f" the task's remaining time {self.task.remaining_time}."
                " Set a larger plan-ahead."
            )
            end_time = current_time + plan_ahead

        # Begin STRL max expression to consider task placements in time.
        start_time = current_time
        while start_time < end_time - self.task.remaining_time:
            # Set up space-time matrix for the start time.
            matrix = SpaceTimeMatrix(
                workers, current_time, end_time, time_discretization
            )
            self._matrices[start_time] = matrix
            # Set up start time indicator.
            start_time_int = start_time.to(EventTime.Unit.US).time
            start_time_indicator = optimizer.addVar(
                vtype=GRB.BINARY, name=f"{self.name}_is_placed_at_{start_time_int}"
            )
            self._start_time_indicators[start_time] = start_time_indicator
            # Begin STRL nCk(workers, 1, current_time, remaining_time, 1)
            # expression to consider task placements on workers.
            # TODO: prune workers that lack the requested resources.
            partition_vars = []
            for worker_idx, worker in workers.items():
                partition_var = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{self.task.unique_name}_placed_on_{worker.name}_start_{start_time_int}",
                )
                matrix[
                    worker_idx, start_time : start_time + self.task.remaining_time
                ] = partition_var
                partition_vars.append(partition_var)
            # Resource demand constraint.
            optimizer.addConstr(
                gp.quicksum(partition_vars) == start_time_indicator,
                name=f"{self.name}_start_{start_time}_resource_demands",
            )
            start_time += time_discretization

        # Select at most 1 of the possible start times.
        optimizer.addConstr(
            gp.quicksum(self._start_time_indicators.values()) <= self._is_placed
        )

        # Populate _placed_on_worker for placement constraints.
        for worker_idx, worker in workers.items():
            # Get partition variables for this specific worker.
            worker_partition_vars = []
            for start_time, matrix in self._matrices.items():
                partition_var = matrix[worker_idx, start_time]
                worker_partition_vars.append(partition_var)
            # Add the placed_on_worker variable.
            placed_on_worker = optimizer.addVar(
                vtype=GRB.BINARY,
                name=f"{self.task.unique_name}_placed_on_{worker.name}",
            )
            optimizer.addConstr(
                placed_on_worker == gp.quicksum(worker_partition_vars),
                name=f"{self.name}_placed_on_{worker.name}",
            )
            self._placed_on_worker[worker_idx] = placed_on_worker


class TetriSched(BaseScheduler):
    """Implements an TetriSched formulation of the scheduling problem for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
        goal (`str`): The goal to use as the optimization objective.
        time_limit (`int`): The time (in seconds) to keep searching for new solutions
            without any changes to either the incumbent or the best bound.
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
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        enforce_deadlines: bool = False,
        retract_schedules: bool = True,
        goal: str = "max_goodput",
        time_limit: int = 20,
        time_discretization: EventTime = EventTime(time=1, unit=EventTime.Unit.US),
        plan_ahead: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(TetriSched, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            _flags=_flags,
        )
        self._goal = goal
        self._gap_time_limit = time_limit  # In seconds.
        self._time_discretization = time_discretization
        self._plan_ahead = plan_ahead
        self._log_to_file = log_to_file
        self._allowed_to_miss_deadlines = set()

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
        if self._log_to_file:
            optimizer.Params.LogFile = (
                f"./gurobi_{current_time.to(EventTime.Unit.US).time}.log"
            )

        # # If the goal is goodput, set the MIPGap to 0.1.
        # if self._goal == "max_goodput":
        #     optimizer.Params.MIPGap = 0.1

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

        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
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
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks for scheduling across {len(workers)} workers. These tasks were: "
            f"{[task.unique_name for task in tasks_to_be_scheduled]}."
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler is also considering the following "
            f"{len(previously_placed_tasks)} for their effects on the current "
            f"placements: {[task.unique_name for task in previously_placed_tasks]}."
        )

        # Construct the model and the variables for each of the tasks.
        scheduler_start_time = time.time()
        placements = []
        if len(tasks_to_be_scheduled) != 0:
            optimizer = self._initialize_optimizer(sim_time)
            tasks_to_variables = self._add_variables(
                sim_time,
                optimizer,
                workload,
                tasks_to_be_scheduled + previously_placed_tasks,
                workers,
            )
            assert all(
                not tasks_to_variables[task.unique_name].previously_placed
                for task in tasks_to_be_scheduled
            ), (
                "The tasks to be scheduled were incorrectly assumed to"
                "be previously scheduled by the Optimizer."
            )
            assert all(
                tasks_to_variables[task.unique_name] for task in previously_placed_tasks
            ), (
                "The previously placed tasks were incorrectly assumed "
                "to be schedulable by the Optimizer."
            )

            # Add the constraints to ensure that dependency constraints are met and
            # resources are not oversubscribed.
            self._add_resource_constraints(
                optimizer, tasks_to_variables, workload, workers
            )

            # Add the objectives and optimize the model.
            self._add_objective(optimizer, tasks_to_variables, workload)
            optimizer.optimize(
                callback=lambda optimizer, where: self._termination_check_callback(
                    sim_time, optimizer, where
                )
            )
            self._logger.debug(
                f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned the "
                f"status {optimizer.status}."
            )

            # Collect the placement results.
            if optimizer.Status == GRB.OPTIMAL or (
                optimizer.Status == GRB.INTERRUPTED and optimizer._solution_found
            ):
                self._logger.debug(
                    f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned "
                    f"the objective value {optimizer.objVal}."
                )
                for task_variables in tasks_to_variables.values():
                    if task_variables.previously_placed:
                        continue
                    task = task_variables.task

                    # Find the Worker and the WorkerPool where the Task was placed.
                    placement_worker_id = None
                    placement_worker_pool_id = None
                    for worker_id, worker in workers.items():
                        if isinstance(
                            task_variables.placed_on_worker(worker_id), gp.Var
                        ):
                            if task_variables.placed_on_worker(worker_id).X == 1:
                                placement_worker_id = worker.id
                                placement_worker_pool_id = worker_to_worker_pool[
                                    worker.id
                                ]

                    # If the task was placed, find the start time.
                    if placement_worker_pool_id is not None:
                        start_time = None
                        for (
                            event_time,
                            indicator,
                        ) in task_variables.start_time_indicators.items():
                            if indicator.X == 1:
                                start_time = event_time
                                break

                        meets_deadline = (
                            start_time + task.remaining_time <= task.deadline
                        )

                        if task_variables.enforce_deadlines and not meets_deadline:
                            self._logger.debug(
                                "[%s] Failed to place %s because the deadline "
                                "could not be met with the suggested start time of %s.",
                                sim_time.to(EventTime.Unit.US).time,
                                task.unique_name,
                                start_time,
                            )
                            placements.append(
                                Placement(
                                    task=task_variables.task,
                                    placement_time=None,
                                    worker_pool_id=None,
                                    worker_id=None,
                                )
                            )
                        else:
                            self._logger.debug(
                                "[%s] Placed %s (with deadline %s and remaining time "
                                "%s) on %s to be started at %s.",
                                sim_time.to(EventTime.Unit.US).time,
                                task.unique_name,
                                task.deadline,
                                task.remaining_time,
                                worker_pools.get_worker_pool(placement_worker_pool_id),
                                start_time,
                            )
                            placements.append(
                                Placement(
                                    task=task_variables.task,
                                    placement_time=start_time,
                                    worker_pool_id=placement_worker_pool_id,
                                    worker_id=placement_worker_id,
                                )
                            )
                    else:
                        self._logger.debug(
                            "[%s] Failed to place %s because no WorkerPool "
                            "could accomodate the resource requirements.",
                            sim_time.to(EventTime.Unit.US).time,
                            task.unique_name,
                        )
                        placements.append(
                            Placement(
                                task=task_variables.task,
                                placement_time=None,
                                worker_pool_id=None,
                                worker_id=None,
                            )
                        )
            else:
                # No feasible solution was found, cancel the tasks that were required
                # to be scheduled, and assume the prior placement for all the other
                # tasks.
                for task in tasks_to_be_scheduled:
                    placements.append(
                        Placement(
                            task=task,
                            placement_time=None,
                            worker_pool_id=None,
                            worker_id=None,
                        )
                    )
                self._logger.warning(f"[{sim_time.time}] Failed to place any task.")
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

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        workload: Workload,
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
        tasks_to_variables = {}
        for task in tasks_to_be_scheduled:
            # Retrieve the TaskGraph for all the tasks that are to be scheduled.
            task_graph = workload.get_task_graph(task.task_graph)
            if task_graph in self._allowed_to_miss_deadlines:
                continue

            # If the Task is not already SCHEDULED / RUNNING, and all of its parents
            # are not in the set of Tasks to be scheduled, then this TaskGraph has
            # suffered a branch misprediction, and should be allowed to miss its
            # deadline.
            all_sources_present = all(
                source in tasks_to_be_scheduled for source in task_graph.get_sources()
            )
            if (
                task.state not in (TaskState.SCHEDULED, TaskState.RUNNING)
                and not all_sources_present
            ):
                self._allowed_to_miss_deadlines.add(task.task_graph)

        self._logger.debug(
            "[%s] The deadlines will not be enforced on TaskGraphs: %s",
            sim_time.to(EventTime.Unit.US).time,
            self._allowed_to_miss_deadlines,
        )

        if self._plan_ahead is not None:
            plan_ahead = sim_time + self._plan_ahead
        else:
            # Get most lax deadline.
            max_deadline = max(map(lambda t: t.deadline, tasks_to_be_scheduled))
            plan_ahead = sim_time
            while plan_ahead < max_deadline:
                plan_ahead += self._time_discretization

        for task in tasks_to_be_scheduled:
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                sim_time,
                plan_ahead,
                self._time_discretization,
                task,
                workers,
                optimizer,
                self.enforce_deadlines,
            )
        return tasks_to_variables

    def _add_resource_constraints(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
        workers: Mapping[int, Worker],
    ) -> None:
        """Generates the variables and constraints to ensure that the resource
        availability inside the Workers is not violated.

        Args:
            current_time (`EventTime`): The time at which the scheduler was
                invoked.
            optimizer (`gp.Model`): The instance of the Gurobi model to which the
                variables and constraints must be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): A mapping
                from the name of the Task to its corresponding variables inside the
                optimizer.
            workload (`Workload`): The workload with which the scheduler was invoked.
            workers (`Mapping[int, Worker]`): A mapping of the unique index of the
                Worker to its instance.
        """
        # Supply: ensure usage <= available resources.
        task = next(iter(tasks_to_variables.values()))
        time_slices = next(iter(task.space_time_matrices.values())).time_slices
        for time_slice in time_slices:
            time_slice_us = time_slice.to(EventTime.Unit.US).time
            for worker_idx, worker in workers.items():
                # Compute total demand placed on a worker.
                total_demand = defaultdict(lambda: 0)
                for task_variables in tasks_to_variables.values():
                    resource_requirements = task_variables.task.resource_requirements
                    task_demand = resource_requirements.get_unique_resource_types()
                    for resource, quantity in task_demand.items():
                        total_demand[
                            resource
                        ] += quantity * task_variables.placed_on_worker(worker_idx)
                worker_supply = worker.resources.get_unique_resource_types()
                for resource, expression in total_demand.items():
                    optimizer.addConstr(
                        expression <= worker_supply.get(resource, 0),
                        name=(
                            f"worker_{worker.name}_t_{time_slice_us}_"
                            f"resource_{resource.name}_constraint"
                        ),
                    )

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        # Construct the reward variables for each of the TaskGraphs were
        # made available for scheduling.
        task_graph_reward_variables = {}
        for task_variable in tasks_to_variables.values():
            task_graph = workload.get_task_graph(task_variable.task.task_graph)
            if task_graph.name not in task_graph_reward_variables:
                task_graph_reward_variables[task_graph.name] = optimizer.addVar(
                    vtype=GRB.INTEGER,
                    name=f"{task_graph.name}_reward",
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                )
        if self._goal == "tetri_sched_naive":
            is_placed_at_time_indicators = [
                task.start_time_indicators.values()
                for task in tasks_to_variables.values()
            ]
            optimizer.setObjective(
                gp.quicksum(itertools.chain(*is_placed_at_time_indicators)),
                sense=GRB.MAXIMIZE,
            )

        elif self._goal == "max_goodput":
            raise NotImplementedError

            for (
                task_graph_name,
                task_graph_reward_variable,
            ) in task_graph_reward_variables.items():
                task_graph = workload.get_task_graph(task_graph_name)

                # Define reward variables for the deepest tasks released during this
                # invocation. A `deep` task is defined as one whose children either
                # do not exist (sink task) or that they were not released during
                # this invocation. If the `release_taskgraphs` option was set, this
                # defaults to checking the status of the sink nodes in a TaskGraph.
                task_reward_variables = []
                for task_variable in tasks_to_variables.values():
                    if task_variable.task.task_graph == task_graph_name:
                        # Check if the task is a sink task.
                        # Either the task is a sink task (i.e., has no children), or
                        # none of its children are included in the currently released
                        # tasks during this invocation of the Scheduler.
                        if self.release_taskgraphs and not task_graph.is_sink_task(
                            task_variable.task
                        ):
                            continue

                        is_sink_task = not any(
                            child.unique_name in tasks_to_variables
                            for child in task_graph.get_children(task_variable.task)
                        )

                        if not self.release_taskgraphs and not is_sink_task:
                            continue

                        # Check if the task is placed.
                        is_placed = optimizer.addVar(
                            vtype=GRB.BINARY,
                            name=f"{task_variable.name}_is_placed_reward",
                        )
                        optimizer.addGenConstrIndicator(
                            is_placed,
                            0,
                            gp.quicksum(task_variable.placed_on_workers),
                            GRB.EQUAL,
                            0,
                            name=f"{task_variable.name}_is_placed_reward_FALSE",
                        )
                        optimizer.addGenConstrIndicator(
                            is_placed,
                            1,
                            gp.quicksum(task_variable.placed_on_workers),
                            GRB.EQUAL,
                            1,
                            name=f"{task_variable.name}_is_placed_reward_TRUE",
                        )

                        # Check if the task meets its deadline.
                        meets_deadline = optimizer.addVar(
                            vtype=GRB.BINARY,
                            name=f"{task_variable.name}_meets_deadline",
                        )
                        optimizer.addGenConstrIndicator(
                            meets_deadline,
                            0,
                            task_variable.start_time
                            + task_variable.task.remaining_time.to(
                                EventTime.Unit.US
                            ).time,
                            GRB.GREATER_EQUAL,
                            task_variable.task.deadline.to(EventTime.Unit.US).time + 1,
                            name=f"{task_variable.name}_meets_deadline_FALSE",
                        )
                        optimizer.addGenConstrIndicator(
                            meets_deadline,
                            1,
                            task_variable.start_time
                            + task_variable.task.remaining_time.to(
                                EventTime.Unit.US
                            ).time,
                            GRB.LESS_EQUAL,
                            task_variable.task.deadline.to(EventTime.Unit.US).time,
                            name=f"{task_variable.name}_meets_deadline_TRUE",
                        )

                        # Compute the reward.
                        task_reward = optimizer.addVar(
                            vtype=GRB.BINARY, name=f"{task_variable.name}_reward"
                        )
                        optimizer.addGenConstrAnd(
                            task_reward,
                            [is_placed, meets_deadline],
                            name=f"{task_variable.name}_reward_constraint",
                        )
                        task_reward_variables.append(task_reward)

                # Now that we have all the rewards for the individual task, compute
                # the final reward as an AND of all these rewards.
                optimizer.addGenConstrAnd(
                    task_graph_reward_variable,
                    task_reward_variables,
                    name=f"{task_graph_name}_reward_constraint",
                )

            optimizer.setObjective(
                gp.quicksum(task_graph_reward_variables.values()), sense=GRB.MAXIMIZE
            )
        else:
            raise ValueError(f"The goal {self._goal} is not supported.")

        # Update the time at which the gap was updated and the gap itself.
        optimizer._solution_found = False
        optimizer._current_gap = float("inf")
        optimizer._last_gap_update = time.time()

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
        if (
            solver_time - optimizer._last_gap_update > self._gap_time_limit
        ) and optimizer._solution_found:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't "
                "changed in %f seconds, and there is a valid solution. "
                "Terminating.",
                sim_time.to(EventTime.Unit.US).time,
                solver_time - optimizer._last_gap_update,
            )
            optimizer.terminate()
        elif solver_time - optimizer._last_gap_update > self._gap_time_limit:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't "
                "changed in %f seconds, and no valid solution has yet been "
                "found. Terminating.",
                sim_time.to(EventTime.Unit.US).time,
                solver_time - optimizer._last_gap_update,
            )
            optimizer.terminate()
