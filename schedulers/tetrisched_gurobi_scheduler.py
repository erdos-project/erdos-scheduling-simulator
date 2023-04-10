import multiprocessing
import sys
import time
from collections import defaultdict
from copy import copy
from typing import Mapping, Optional, Sequence, Union

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import Placement, Placements, Task, TaskState, Workload


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
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        task: Task,
        workers: Mapping[int, Worker],
        optimizer: gp.Model,
        enforce_deadlines: bool = True,
    ):
        # TODO: implement lookahead.
        self._task = task
        self._enforce_deadlines = enforce_deadlines
        self._start_time = None

        # Placement characteristics
        # Indicator variables which signify the task's start time.
        self._start_time_indicators = {}

        # Maps worker_idx, start_time to a partition variable indicating whether the
        # task will begin executing on the worker at the start_time.
        self._partition_vars = {worker_idx: {} for worker_idx in workers}

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
            self._meets_deadline = int(
                current_time + task.remaining_time <= task.deadline
            )

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
            self._partition_vars[previously_placed_worker][current_time] = 1
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
                # Add constraint that the task must be placed.
                optimizer.addConstr(
                    self.is_placed == 1, name=f"{task.unique_name}_must_be_placed"
                )

                # Set up variables to signify if the Task was placed on a Worker.
                previously_placed_worker = task.current_placement.worker_id
                previously_placed_worker_idx = -1
                for worker_idx, worker in workers.items():
                    if worker.id == previously_placed_worker:
                        self._placed_on_worker[worker_idx].Start = 1
                        previously_placed_worker_idx = worker_idx
                    else:
                        self._placed_on_worker[worker_idx].Start = 0

                # Seed expected start time.
                for (
                    start_time,
                    start_time_indicator,
                ) in self._start_time_indicators.items():
                    if (
                        start_time <= task.expected_start_time
                        and task.expected_start_time < start_time + time_discretization
                    ):
                        start_time_indicator.Start = 1
                    else:
                        start_time_indicator.Start = 0

                # Seed partition variables.
                for worker_idx, time_to_partition_var in self._partition_vars.items():
                    for start_time, partition_var in time_to_partition_var.items():
                        if (
                            worker_idx == previously_placed_worker_idx
                            and start_time <= task.expected_start_time
                            and task.expected_start_time
                            < start_time + time_discretization
                        ):
                            partition_var.Start = 1
                        else:
                            partition_var.Start = 0

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
    def start_time_indicators(self) -> Mapping[EventTime, Union[int, gp.Var]]:
        """Mapping from start time to indicator whether the start time is chosen."""
        return self._start_time_indicators

    @property
    def is_placed(self) -> Union[int, gp.Var]:
        """Check if the scheduler found a placement for the task."""
        return self._is_placed

    @property
    def meets_deadline(self) -> Union[int, gp.LinExpr]:
        """Check whether the scheduler found a placement that meets the task's
        deadline."""
        return self._meets_deadline

    def partition_variables_at(
        self,
        worker_id: int,
        event_time: EventTime,
    ) -> Sequence[Union[int, gp.Var]]:
        """Returns all partition variables for the worker at the event time."""
        partition_vars = []
        time_to_partition_vars = self._partition_vars.get(worker_id)
        for start_time, partition_var in time_to_partition_vars.items():
            if (
                start_time <= event_time
                and event_time < start_time + self.task.remaining_time
            ):
                partition_vars.append(partition_var)

        return partition_vars

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
        latest_start_time = current_time + plan_ahead - time_discretization
        if enforce_deadlines:
            latest_start_time = min(
                latest_start_time, self.task.deadline - self.task.remaining_time
            )

        # Begin STRL max expression to consider task placements in time.
        start_time = current_time
        while start_time <= latest_start_time:
            # Set up start time indicator.
            start_time_int = start_time.to(EventTime.Unit.US).time
            start_time_indicator = optimizer.addVar(
                vtype=GRB.BINARY, name=f"{self.name}_is_placed_at_{start_time_int}"
            )
            self._start_time_indicators[start_time] = start_time_indicator
            # Begin STRL nCk(workers, 1, current_time, remaining_time, 1)
            # expression to consider task placements on workers.

            partition_vars = []
            for worker_idx, worker in workers.items():
                compatible_strategies = worker.get_compatible_strategies(
                    self.task.available_execution_strategies
                )
                if len(compatible_strategies) == 0:
                    partition_var = 0
                else:
                    partition_var = optimizer.addVar(
                        vtype=GRB.BINARY,
                        name=(
                            f"{self.task.unique_name}_placed_on_{worker.name}"
                            "_start_{start_time_int}"
                        ),
                    )
                self._partition_vars[worker_idx][start_time] = partition_var
                partition_vars.append(partition_var)
            # Worker demand constraint.
            optimizer.addConstr(
                gp.quicksum(partition_vars) == start_time_indicator,
                name=f"{self.name}_start_{start_time}_worker_demands",
            )
            start_time += time_discretization

        # Select at most 1 of the possible start times.
        optimizer.addConstr(
            gp.quicksum(self._start_time_indicators.values()) == self._is_placed
        )

        # Populate _placed_on_worker for placement constraints.
        for worker_idx, worker in workers.items():
            # Add the placed_on_worker variable.
            placed_on_worker = optimizer.addVar(
                vtype=GRB.BINARY,
                name=f"{self.task.unique_name}_placed_on_{worker.name}",
            )
            optimizer.addConstr(
                placed_on_worker
                == gp.quicksum(self._partition_vars[worker_idx].values()),
                name=f"{self.name}_placed_on_{worker.name}",
            )
            self._placed_on_worker[worker_idx] = placed_on_worker

        # Indicate whether the task meets its deadline.
        meets_deadline_indicators = []
        for start_time, meets_deadline_indicator in self.start_time_indicators.items():
            if start_time + self.task.remaining_time <= self.task.deadline:
                meets_deadline_indicators.append(meets_deadline_indicator)
        self._meets_deadline = gp.quicksum(meets_deadline_indicators)


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
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        enforce_deadlines: bool = False,
        retract_schedules: bool = True,
        goal: str = "max_goodput",
        time_limit: EventTime = EventTime(time=20, unit=EventTime.Unit.S),
        time_discretization: EventTime = EventTime(time=1, unit=EventTime.Unit.US),
        plan_ahead: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(TetriSchedGurobiScheduler, self).__init__(
            runtime=runtime,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            _flags=_flags,
        )
        self._goal = goal
        self._gap_time_limit = time_limit
        self._time_discretization = time_discretization
        self._plan_ahead = plan_ahead
        self._log_to_file = log_to_file

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

            # Compute plan-ahead.
            if self._plan_ahead > EventTime.zero():
                plan_ahead = sim_time + self._plan_ahead
            else:
                # Get most lax deadline.
                max_deadline = max(
                    tasks_to_be_scheduled, key=lambda t: t.deadline
                ).deadline
                max_remaining_time = max(
                    tasks_to_be_scheduled, key=lambda t: t.remaining_time
                ).remaining_time
                plan_ahead = sim_time
                while plan_ahead < max(max_deadline, sim_time + max_remaining_time):
                    plan_ahead += self._time_discretization
            self._logger.debug(
                "[%s] Setting plan-ahead to %s",
                sim_time.to(EventTime.Unit.US).time,
                plan_ahead,
            )

            tasks_to_variables = self._add_variables(
                sim_time,
                plan_ahead,
                optimizer,
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
                sim_time, plan_ahead, optimizer, tasks_to_variables, workers
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
                    placement_worker: Optional[Worker] = None
                    for worker_id, worker in workers.items():
                        if isinstance(
                            task_variables.placed_on_worker(worker_id), gp.Var
                        ):
                            if task_variables.placed_on_worker(worker_id).X == 1:
                                placement_worker = worker

                    # If the task was placed, find the start time.
                    if placement_worker:
                        start_time = None
                        for (
                            event_time,
                            indicator,
                        ) in task_variables.start_time_indicators.items():
                            if indicator.X == 1:
                                start_time = event_time
                                break

                        self._logger.debug(
                            "[%s] Placed %s (with deadline %s and remaining time "
                            "%s) on %s to be started at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            task.unique_name,
                            task.deadline,
                            task.remaining_time,
                            worker_pools.get_worker_pool(
                                worker_to_worker_pool[placement_worker.id]
                            ),
                            start_time,
                        )
                        execution_strategy = placement_worker.get_compatible_strategies(
                            task_variables.task.available_execution_strategies
                        ).get_fastest_strategy()
                        placements.append(
                            Placement(
                                task=task_variables.task,
                                placement_time=start_time,
                                worker_pool_id=worker_to_worker_pool[
                                    placement_worker.id
                                ],
                                worker_id=placement_worker.id,
                                execution_strategy=execution_strategy,
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
                                execution_strategy=None,
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
                            execution_strategy=None,
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
        plan_ahead: EventTime,
        optimizer: gp.Model,
        tasks_to_be_scheduled: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
        """Generates the variables for the optimization problem.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            plan_ahead (`EventTime`): The absolute maximum time of the time frame
                considered by the scheduler.
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
        sim_time: EventTime,
        plan_ahead: EventTime,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workers: Mapping[int, Worker],
    ) -> None:
        """Generates the variables and constraints to ensure that the resource
        availability inside the Workers is not violated.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            plan_ahead (`EventTime`): The absolute maximum time of the time frame
                considered by the scheduler.
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
        current_time = sim_time
        while current_time < plan_ahead:
            current_time_us = current_time.to(EventTime.Unit.US).time
            for worker_idx, worker in workers.items():
                # Compute total demand placed on a worker.
                total_demand = defaultdict(lambda: 0)
                for task_variables in tasks_to_variables.values():
                    compatible_strategies = worker.get_compatible_strategies(
                        task_variables.task.available_execution_strategies
                    )
                    if len(compatible_strategies) == 0:
                        # There are no compatible strategies for this (Task, Worker)
                        # combination. Hence, this Task cannot affect the demand of a
                        # resource on this Worker. Skipping.
                        continue
                    resource_requirements = (
                        compatible_strategies.get_fastest_strategy().resources
                    )
                    task_demand = resource_requirements.get_unique_resource_types()
                    for resource, quantity in task_demand.items():
                        partition_vars = task_variables.partition_variables_at(
                            worker_idx, current_time
                        )
                        total_demand[resource] += quantity * gp.quicksum(partition_vars)
                worker_supply = worker.resources.get_unique_resource_types()
                for resource, expression in total_demand.items():
                    optimizer.addConstr(
                        expression <= worker_supply.get(resource, 0),
                        name=(
                            f"worker_{worker.name}_t_{current_time_us}_"
                            f"resource_{resource.name}_constraint"
                        ),
                    )
            current_time += self._time_discretization

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        if self._goal == "tetri_sched_naive":
            is_placed_indicators = [
                task.is_placed for task in tasks_to_variables.values()
            ]
            optimizer.setObjective(
                gp.quicksum(is_placed_indicators),
                sense=GRB.MAXIMIZE,
            )
        elif self._goal == "max_goodput":
            task_reward_variables = []
            for task_variable in tasks_to_variables.values():
                # Check if the task meets its deadline.
                meets_deadline = optimizer.addVar(
                    vtype=GRB.BINARY,
                    name=f"{task_variable.name}_meets_deadline",
                )
                optimizer.addConstr(
                    meets_deadline == task_variable.meets_deadline,
                    name=f"{task_variable.name}_meets_deadline_constraint",
                )
                # Compute the reward.
                task_reward = optimizer.addVar(
                    vtype=GRB.BINARY, name=f"{task_variable.name}_reward"
                )
                if isinstance(task_variable.is_placed, gp.Var):
                    optimizer.addGenConstrAnd(
                        task_reward,
                        [task_variable.is_placed, meets_deadline],
                    )
                elif task_variable.is_placed == 0:
                    optimizer.addConstr(
                        task_reward == 0,
                    )
                elif task_variable.is_placed == 1:
                    optimizer.addConstr(
                        task_reward == meets_deadline,
                    )
                else:
                    raise ValueError(
                        "Unexpected value for TaskOptimizerVariables.is_placed"
                    )
                task_reward_variables.append(task_reward)

            optimizer.setObjective(
                gp.quicksum(task_reward_variables), sense=GRB.MAXIMIZE
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
