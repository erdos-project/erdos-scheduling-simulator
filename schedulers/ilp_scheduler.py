import multiprocessing
import sys
import time
from copy import copy, deepcopy
from typing import List, Mapping, Optional, Sequence, Union

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    BranchPredictionPolicy,
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
)


class TaskOptimizerVariables:
    """TaskOptimizerVariables is used to represent the optimizer variables for
    every particular task to be scheduled by the Scheduler.

    The initialization of this instance sets up the basic task-only constraints
    required by the problem.

    Args:
        current_time (`EventTime`): The time at which the scheduler was invoked.
            This is used to set a lower bound on the placement time of the tasks.
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
        task: Task,
        workers: Mapping[int, Worker],
        optimizer: gp.Model,
        enforce_deadlines: bool = True,
        retract_schedules: bool = False,
    ):
        self._task = task
        self._enforce_deadlines = enforce_deadlines

        # Placement characteristics
        # Set up individual variables to signify where the task is placed and with what
        # execution strategy.
        self._placed_on_worker_with_strategy = {
            (worker_id, execution_strategy): 0
            for worker_id in workers.keys()
            for execution_strategy in self._task.available_execution_strategies
        }

        # Timing characteristics.
        if self._task.state == TaskState.RUNNING:
            # The task is already running, set the start time to the current
            # simulation time, since we use the remaining time to count the
            # time at which this Task will relinquish its resources.
            self._previously_placed = True
            self._start_time = current_time.to(EventTime.Unit.US).time

            # Find the Worker where the Task was previously placed and add
            # constraints to inform the scheduler of its placement.
            if (
                self._task.current_placement is None
                or self._task.current_placement.worker_id is None
            ):
                raise ValueError(
                    f"Task {self._task.unique_name} in state {self._task.state} does "
                    f"not have a cached prior Placement or the Worker ID is empty."
                )

            placed_key = (
                self.__get_worker_index_from_previous_placement(self._task, workers),
                self._task.current_placement.execution_strategy,
            )
            self._placed_on_worker_with_strategy[placed_key] = 1
        else:
            # The task is not currently running, we initialize variables to let the
            # optimizer decide the start time of the task, along with the worker where
            # its going to be placed and the execution strategy to run the task with.
            self._previously_placed = False
            self._start_time = optimizer.addVar(
                lb=max(
                    current_time.to(EventTime.Unit.US).time + 1,
                    task.release_time.to(EventTime.Unit.US).time,
                ),
                vtype=GRB.INTEGER,
                name=f"{task.unique_name}_start",
            )

            schedulable_worker_to_strategies: Mapping[int, ExecutionStrategies] = {}
            for worker_id, worker in workers.items():
                cleared_worker = deepcopy(worker)
                compatible_strategies = cleared_worker.get_compatible_strategies(
                    self._task.available_execution_strategies
                )
                if len(compatible_strategies) != 0:
                    schedulable_worker_to_strategies[worker_id] = compatible_strategies

            for (
                worker_id,
                execution_strategy,
            ) in self._placed_on_worker_with_strategy.keys():
                if (
                    worker_id in schedulable_worker_to_strategies.keys()
                    and execution_strategy
                    in schedulable_worker_to_strategies[worker_id]
                ):
                    self._placed_on_worker_with_strategy[
                        (worker_id, execution_strategy)
                    ] = optimizer.addVar(
                        vtype=GRB.BINARY,
                        name=f"{task.unique_name}_placed_on_{workers[worker_id].name}_"
                        f"with_batch_size_{execution_strategy.batch_size}_runtime_"
                        f"{execution_strategy.runtime.to(EventTime.Unit.US).time}",
                    )

            # If the task was previously SCHEDULED, we seed the previous placement time
            # and strategies to help the optimizer in making a decision.
            if task.state == TaskState.SCHEDULED:
                self._start_time.Start = task.expected_start_time.to(
                    EventTime.Unit.US
                ).time
                placed_key = (
                    self.__get_worker_index_from_previous_placement(
                        self._task, workers
                    ),
                    self._task.current_placement.execution_strategy,
                )
                for (
                    placement_key,
                    placement_variable,
                ) in self._placed_on_worker_with_strategy.items():
                    if placement_key == placed_key:
                        placement_variable.Start = 1
                    else:
                        placement_variable.Start = 0

            # Initialize the constraints for the variables.
            self.initialize_constraints(optimizer, enforce_deadlines, retract_schedules)

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
        """Returns a Boolean indicating whether the task denoted by this instance
        has been previously placed or not."""
        return self._previously_placed

    @property
    def enforce_deadlines(self) -> bool:
        return self._enforce_deadlines

    def placed_on_worker_with_strategy(
        self, worker_id: int, execution_strategy: ExecutionStrategy
    ) -> Optional[Union[int, gp.Var]]:
        """Check if the Task was placed on a particular Worker with the given execution
        strategy.

        Args:
            worker_id (`int`): The ID of the Worker to check.
            execution_strategy (`ExecutionStrategy`): The execution strategy to check.

        Returns:
            The method has the following return values:
                - `None`: If the `Worker` ID was not registered with the instance.
                - `int`: If the `Task` was forced to not be placed on this `Worker`.
                - `gp.Var`: A Gurobi variable representing the placement solution.
        """
        return self._placed_on_worker_with_strategy.get((worker_id, execution_strategy))

    @property
    def placed_on_workers(self) -> Sequence[Union[int, gp.Var]]:
        """Retrieves the binary Gurobi variables representing the placement of the
        task on the collection of `Worker`s registered with the instance."""
        return self._placed_on_worker_with_strategy.values()

    def __str__(self) -> str:
        return f"TaskOptimizerVariables(name={self.name})"

    def __repr__(self) -> str:
        return str(self)

    def _initialize_timing_constraints(
        self,
        optimizer: gp.Model,
        enforce_deadlines: bool,
    ) -> None:
        # Add a constraint to ensure that if enforcing deadlines is required, then
        # we are setting the start time to something that meets the deadline.
        if enforce_deadlines:
            deadline_enforcement_expression = gp.LinExpr(self.start_time)
            for (
                placement_key,
                placement_variable,
            ) in self._placed_on_worker_with_strategy.items():
                _, execution_strategy = placement_key
                deadline_enforcement_expression.add(
                    placement_variable
                    * execution_strategy.runtime.to(EventTime.Unit.US).time
                )
            optimizer.addConstr(
                deadline_enforcement_expression
                <= self.task.deadline.to(EventTime.Unit.US).time,
                name=f"{self.name}_enforce_deadlines",
            )

    def _initialize_placement_constraints(
        self, optimizer: gp.Model, retract_schedules: bool
    ) -> None:
        # Add a constraint to ensure that the task is only placed on a single Worker.
        # We constrain the sum of the individual indicator variables for the placement
        # on a specific Worker to be at most 1.
        # A sum of 0 implies that the task was not placed on any Worker.
        if self.task.state == TaskState.SCHEDULED and not retract_schedules:
            optimizer.addConstr(
                gp.quicksum(self._placed_on_worker_with_strategy.values()) == 1,
                name=f"{self.name}_previously_scheduled_required_placement",
            )
        else:
            optimizer.addConstr(
                gp.quicksum(self._placed_on_worker_with_strategy.values()) <= 1,
                name=f"{self.name}_consistent_placement",
            )

    def initialize_constraints(
        self,
        optimizer: gp.Model,
        enforce_deadlines: bool,
        retract_schedules: bool,
    ) -> None:
        """Initializes the constraints for the particular `Task`.

        Args:
            optimizer (`gp.Model`): The Gurobi model to which the constraints must
                be added.
        """
        self._initialize_timing_constraints(optimizer, enforce_deadlines)
        self._initialize_placement_constraints(optimizer, retract_schedules)

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


class ILPScheduler(BaseScheduler):
    """Implements an ILP formulation of the scheduling problem for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
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
        goal (`str`): The goal to use as the optimization objective.
        time_limit (`EventTime`): The time (in seconds) to keep searching for new
            solutions without any changes to either the incumbent or the best bound.
        log_to_file (`bool`): If `True`, the scheduler writes the Gurobi search
            log to files with the format "gurobi_{sim_time}.log".
        _flags (`Optional[absl.flags]`): The runtime flags that are used to initialize
            a logger instance.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        lookahead: EventTime = EventTime(time=0, unit=EventTime.Unit.US),
        enforce_deadlines: bool = False,
        policy: BranchPredictionPolicy = BranchPredictionPolicy.RANDOM,
        branch_prediction_accuracy: float = 0.50,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        goal: str = "max_goodput",
        batching: bool = False,
        time_limit: EventTime = EventTime(20, EventTime.Unit.S),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        if not enforce_deadlines and goal == "max_goodput":
            raise ValueError(
                f"The goal {goal} cannot be optimized when "
                f"deadline enforcement is {enforce_deadlines}."
            )
        super(ILPScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            policy=policy,
            branch_prediction_accuracy=branch_prediction_accuracy,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        self._goal = goal
        self._batching = batching
        self._gap_time_limit = time_limit.to(EventTime.Unit.S).time
        self._log_to_file = log_to_file
        self._log_times = set(map(int, _flags.scheduler_log_times)) if _flags else set()
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
        optimizer = gp.Model("ILPScheduler")

        # Don't log the output to the console, instead log it to a file.
        optimizer.Params.LogToConsole = 0
        if (
            self._log_to_file
            or current_time.to(EventTime.Unit.US).time in self._log_times
        ):
            optimizer.Params.LogFile = (
                f"./gurobi_{current_time.to(EventTime.Unit.US).time}.log"
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
            release_taskgraphs=self.release_taskgraphs,
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
        workers: Mapping[int, Worker] = {}
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
            self._add_task_dependency_constraints(
                optimizer, tasks_to_variables, workload, workers
            )
            self._add_resource_constraints(
                optimizer, tasks_to_variables, workload, workers
            )

            # Add the objectives and optimize the model.
            self._add_objective(optimizer, tasks_to_variables, workload, workers)

            if self._log_to_file or (
                sim_time.to(EventTime.Unit.US).time in self._log_times
            ):
                optimizer.write(f"./gurobi_{sim_time.to(EventTime.Unit.US).time}.lp")
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

                # Write the solution to the SOL file, if requested.
                if self._log_to_file or (
                    sim_time.to(EventTime.Unit.US).time in self._log_times
                ):
                    optimizer.write(
                        f"./gurobi_{sim_time.to(EventTime.Unit.US).time}.sol"
                    )

                for task_variables in tasks_to_variables.values():
                    if task_variables.previously_placed:
                        continue
                    task = task_variables.task
                    # Find the starting time of the Task.
                    assert type(task_variables.start_time) == gp.Var, (
                        f"Incorrect type retrieved for start time of "
                        f"{task.unique_name}: {type(task_variables.start_time)}"
                    )
                    start_time = EventTime(
                        int(task_variables.start_time.X), EventTime.Unit.US
                    )

                    # Find the Worker and the WorkerPool where the Task was placed.
                    placement_worker = None
                    placement_strategy = None
                    for worker_id, worker in workers.items():
                        for (
                            strategy
                        ) in task_variables.task.available_execution_strategies:
                            if isinstance(
                                task_variables.placed_on_worker_with_strategy(
                                    worker_id, strategy
                                ),
                                gp.Var,
                            ):
                                if (
                                    task_variables.placed_on_worker_with_strategy(
                                        worker_id, strategy
                                    ).X
                                    == 1
                                ):
                                    placement_worker = worker
                                    placement_strategy = strategy
                                    break

                    # Check if all the tasks from this TaskGraph were placed.
                    all_tasks_placed = True
                    task_not_placed = None
                    task_graph = workload.get_task_graph(task.task_graph)
                    for other_task in task_graph.get_nodes():
                        if other_task.unique_name in tasks_to_variables:
                            placed_on_worker = []
                            for val in tasks_to_variables[
                                other_task.unique_name
                            ].placed_on_workers:
                                if isinstance(val, gp.Var):
                                    placed_on_worker.append(val.X)
                                else:
                                    placed_on_worker.append(val)
                            if sum(placed_on_worker) == 0:
                                all_tasks_placed = False
                                task_not_placed = other_task.unique_name
                                break

                    # If the task was placed, find the start time.
                    if placement_worker and placement_strategy:
                        meets_deadline = (
                            start_time + placement_strategy.runtime <= task.deadline
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
                                Placement.create_task_placement(
                                    task=task_variables.task,
                                    placement_time=None,
                                    worker_pool_id=None,
                                    worker_id=None,
                                    execution_strategy=None,
                                )
                            )
                        elif self.release_taskgraphs and not all_tasks_placed:
                            self._logger.debug(
                                "[%s] Failed to place %s because the task %s from the "
                                "TaskGraph %s could not be placed.",
                                sim_time.to(EventTime.Unit.US).time,
                                task.unique_name,
                                task_not_placed,
                                task.task_graph,
                            )
                            placements.append(
                                Placement.create_task_placement(
                                    task=task_variables.task,
                                    placement_time=None,
                                    worker_pool_id=None,
                                    worker_id=None,
                                    execution_strategy=None,
                                )
                            )
                        else:
                            self._logger.debug(
                                "[%s] Placed %s (with deadline %s and remaining time "
                                "%s) on WorkerPool(%s) to be started at %s.",
                                sim_time.to(EventTime.Unit.US).time,
                                task.unique_name,
                                task.deadline,
                                strategy.runtime,
                                worker_to_worker_pool[placement_worker.id],
                                start_time,
                            )
                            placements.append(
                                Placement.create_task_placement(
                                    task=task_variables.task,
                                    placement_time=start_time,
                                    worker_pool_id=worker_to_worker_pool[
                                        placement_worker.id
                                    ],
                                    worker_id=placement_worker.id,
                                    execution_strategy=strategy,
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
                            Placement.create_task_placement(
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
            workload (`Workload`): The workload definition provided to this invocation
                of the scheduler.
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

        for task in tasks_to_be_scheduled:
            enforce_deadlines = self.enforce_deadlines
            if (
                self.release_taskgraphs
                and task.task_graph in self._allowed_to_miss_deadlines
            ):
                enforce_deadlines = False
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                sim_time,
                task,
                workers,
                optimizer,
                enforce_deadlines,
                self.retract_schedules,
            )
        return tasks_to_variables

    def _add_task_dependency_constraints(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
        workers: Mapping[int, Worker],
    ) -> None:
        """Generates the variables and constraints to ensure that the dependencies
        due to a TaskGraph are respected.

        Args:
            optimizer (`gp.Model`): The instance of the Gurobi model to which the
                variables and constraints must be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): A mapping
                from the name of the Task to its corresponding variables inside the
                optimizer.
            workload (`Workload`): The workload with which the scheduler was invoked.
            workers (`Mapping[int, Worker]`): A mapping of the unique index of the
                Worker to its instance.
        """
        for task_name, variable in tasks_to_variables.items():
            if variable.previously_placed:
                # The task was previously placed, we should not add any constraints
                # for this Task.
                continue

            # Retrieve the variables for all the parents of this Task.
            task_graph = workload.get_task_graph(variable.task.task_graph)
            parent_variables = [
                tasks_to_variables[parent.unique_name]
                for parent in task_graph.get_parents(variable.task)
                if parent.unique_name in tasks_to_variables
            ]

            # Ensure that the task is only placed if all of its parents are placed,
            # and that it is started after the last parent has finished.
            if len(parent_variables) > 0:
                all_parents_placed = optimizer.addVar(
                    vtype=GRB.BINARY, name=f"{task_name}_all_parents_placed"
                )
                parent_placements = []
                for parent_variable in parent_variables:
                    for worker_id, worker in workers.items():
                        for (
                            strategy
                        ) in parent_variable.task.available_execution_strategies:
                            optimizer.addConstr(
                                variable.start_time
                                >= parent_variable.start_time
                                + parent_variable.placed_on_worker_with_strategy(
                                    worker_id, strategy
                                )
                                * (strategy.runtime.to(EventTime.Unit.US).time + 1),
                                name=f"{task_name}_start_after_{parent_variable.name}_"
                                f"on_worker_{worker.name}_with_batch_size_"
                                f"{strategy.batch_size}_runtime_"
                                f"{strategy.runtime.to(EventTime.Unit.US).time}",
                            )
                    parent_placements.extend(parent_variable.placed_on_workers)

                # Construct an indicator variable that checks if all the parents were
                # placed on some worker or not.
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    0,
                    gp.quicksum(parent_placements),
                    GRB.LESS_EQUAL,
                    len(parent_variables) - 1,
                    name=f"{task_name}_parents_placed_False",
                )
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    1,
                    gp.quicksum(parent_placements),
                    GRB.EQUAL,
                    len(parent_variables),
                    name=f"{task_name}_parents_placed_True",
                )

                # If all of the parents were not placed, then we cannot place this task.
                # Otherwise, we allow the optimizer to choose when to place the task.
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    0,
                    gp.quicksum(variable.placed_on_workers),
                    GRB.EQUAL,
                    0,
                    name=f"{task_name}_placement_False",
                )

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
        # Construct variables for overlaps with all the other tasks corresponding to
        # each task.
        task_pairs = []
        for task_1_name, task_1_variable in tasks_to_variables.items():
            for task_2_name in tasks_to_variables.keys():
                if task_1_name != task_2_name:
                    task_pairs.append((task_1_name, task_2_name))
        task_pair_overlap_variables = optimizer.addVars(
            task_pairs, vtype=GRB.BINARY, name="Overlap"
        )
        optimizer.update()

        # For each task, add constraints to the Overlap variable that ensures that
        # it is set to 1 if the task's start time overlaps with the execution of the
        # other task, and 0 otherwise.
        for task_1_name, task_1_variable in tasks_to_variables.items():
            for task_2_name, task_2_variable in tasks_to_variables.items():
                if task_1_name == task_2_name:
                    continue

                # Find the overlap variable for this pair.
                overlap_variables = task_pair_overlap_variables.select(
                    task_1_name, task_2_name
                )
                if len(overlap_variables) != 1:
                    raise ValueError(
                        f"Expected only one Overlap variable for ({task_1_name}, "
                        f"{task_2_name}). Found {len(overlap_variables)}."
                    )
                task_pair_overlap_variable = overlap_variables[0]

                # If the tasks belong to the same graph, then they need to be checked
                # for overlap, otherwise, the two tasks can always overlap.
                if task_1_variable.task.task_graph == task_2_variable.task.task_graph:
                    task_graph = workload.get_task_graph(
                        task_1_variable.task.task_graph
                    )
                    if task_graph.are_dependent(
                        task_1_variable.task, task_2_variable.task
                    ):
                        # If the tasks are dependent on each other,
                        # they can never overlap.
                        optimizer.addConstr(
                            task_pair_overlap_variable == 0,
                            name=f"{task_1_name}_no_overlap_{task_2_name}_dependent",
                        )
                    else:
                        # If the tasks are not dependent on each other,
                        # they may overlap.
                        self._overlaps(
                            optimizer,
                            task_1_variable,
                            task_2_variable,
                            workers,
                            task_pair_overlap_variable,
                        )
                else:
                    self._overlaps(
                        optimizer,
                        task_1_variable,
                        task_2_variable,
                        workers,
                        task_pair_overlap_variable,
                    )

        # Now, for each task, we ensure that the cumulative sum of the resource
        # requirements for all the tasks that overlap with this one doesn't end up
        # oversubscribing any Worker's resources.
        for task_1_name, task_1_variable in tasks_to_variables.items():
            task_dependencies: List[str, TaskOptimizerVariables] = []
            for task_2_name, task_2_variable in tasks_to_variables.items():
                if task_1_name != task_2_name:
                    task_dependencies.append((task_2_name, task_2_variable))

            # For each Worker and each of its resources, ensure no oversubscription.
            for worker_index, worker in workers.items():
                if task_1_variable.previously_placed and all(
                    task_1_variable.placed_on_worker_with_strategy(
                        worker_index, strategy
                    )
                    == 0
                    for strategy in task_1_variable.task.available_execution_strategies
                ):
                    # The task was previously placed, but not on this worker. Thus,
                    # we should skip adding the constraints here.
                    continue

                for (
                    resource,
                    quantity,
                ) in worker.resources.get_unique_resource_types().items():
                    resource_constraint_expression = gp.QuadExpr()
                    # We check each of the strategies for the Task to see which one
                    # has requirements for this resource, and use its placement variable
                    # to check for resource oversubscription.
                    for (
                        execution_strategy
                    ) in task_1_variable.task.available_execution_strategies:
                        task_1_request_for_resource = (
                            execution_strategy.resources.get_total_quantity(resource)
                        )
                        if task_1_request_for_resource == 0:
                            # We ensure earlier that the Worker has enough space to
                            # accomodate each task. If the task does not need this
                            # resource, then we can skip the addition of a constraint
                            # since the task would not have taken up this resource.
                            continue

                        resource_constraint_expression.add(
                            task_1_variable.placed_on_worker_with_strategy(
                                worker_index, execution_strategy
                            )
                            * task_1_request_for_resource
                        )

                    # If the dependency overlaps and the dependency is placed on this
                    # worker, then ensure that the resource type is not oversubscribed.
                    for task_2_name, task_2_variable in task_dependencies:
                        if task_2_variable.previously_placed and all(
                            task_2_variable.placed_on_worker_with_strategy(
                                worker_index, strategy
                            )
                            == 0
                            for strategy in (
                                task_2_variable.task.available_execution_strategies
                            )
                        ):
                            # The task was previously placed, but not on this worker.
                            # Thus, any overlap of these two tasks will not affect the
                            # oversubscription of resources here.
                            continue

                        overlap_variable = task_pair_overlap_variables[
                            (task_1_name, task_2_name)
                        ]
                        for (
                            execution_strategy
                        ) in task_2_variable.task.available_execution_strategies:
                            task_2_resource_reqs = (
                                execution_strategy.resources.get_total_quantity(
                                    resource
                                )
                            )
                            if task_2_resource_reqs == 0:
                                # The second Task has no requirement for this Resource
                                # so its placement will not affect oversubscription, we
                                # can skip adding it to the constraint.
                                continue

                            resource_constraint_expression.add(
                                task_2_variable.placed_on_worker_with_strategy(
                                    worker_index, execution_strategy
                                )
                                * overlap_variable
                                * task_2_resource_reqs
                            )

                    # Add the constraint to the optimizer.
                    optimizer.addConstr(
                        resource_constraint_expression <= quantity,
                        name=f"{task_1_name}_{worker.name}_{resource.name}_constraint",
                    )

    def _overlaps(
        self,
        optimizer: gp.Model,
        task_1: TaskOptimizerVariables,
        task_2: TaskOptimizerVariables,
        workers: Mapping[int, Worker],
        overlap_variable: Optional[gp.Var] = None,
    ) -> gp.Var:
        """Insert an indicator variable that specifies if the two tasks overlap in
        their execution.

        The method checks if either tasks start / end before the other, and assumes
        that they overlap otherwise.

        Args:
            optimizer (`gp.Model`): The model to which the variables and constraints
                must be attached.
            task_1 (`TaskOptimizerVariables`): The optimizer variables associated
                with the first task.
            task_2 (`TaskOptimizerVariables`): The optimizer variables associated
                with the second task.
            overlap_variable (`Optional[gp.Var]`): If provided, the overlap variable
                is used instead of defining and returning a new variable.

        Returns
            A `gp.Var` whose post-solution value specifies if the two tasks overlap.
        """
        # Add an indicator variable that checks if the first task starts after the
        # second task ends.
        task_1_starts_after_task_2_ends = optimizer.addVar(
            vtype=GRB.BINARY, name=f"{task_1.name}_starts_after_{task_2.name}_ends"
        )
        task_2_remaining_time_expr = gp.LinExpr()
        for worker_id in workers.keys():
            for strategy in task_2.task.available_execution_strategies:
                task_2_remaining_time_expr.add(
                    task_2.placed_on_worker_with_strategy(worker_id, strategy)
                    * strategy.runtime.to(EventTime.Unit.US).time
                )
        optimizer.addGenConstrIndicator(
            task_1_starts_after_task_2_ends,
            0,
            task_1.start_time - task_2.start_time - task_2_remaining_time_expr,
            GRB.LESS_EQUAL,
            0,
            name=f"{task_1.name}_starts_after_{task_2.name}_ends_False",
        )
        optimizer.addGenConstrIndicator(
            task_1_starts_after_task_2_ends,
            1,
            task_1.start_time - task_2.start_time - task_2_remaining_time_expr,
            GRB.GREATER_EQUAL,
            1,
            name=f"{task_1.name}_starts_after_{task_2.name}_ends_True",
        )

        # Add an indicator variable that checks if the first task starts before the
        # second task starts.
        task_1_ends_before_task_2_starts = optimizer.addVar(
            vtype=GRB.BINARY, name=f"{task_1.name}_ends_before_{task_2.name}_starts"
        )
        task_1_remaining_time_expr = gp.LinExpr()
        for worker_id in workers.keys():
            for strategy in task_1.task.available_execution_strategies:
                task_1_remaining_time_expr.add(
                    task_1.placed_on_worker_with_strategy(worker_id, strategy)
                    * strategy.runtime.to(EventTime.Unit.US).time
                )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            0,
            task_1.start_time + task_1_remaining_time_expr - task_2.start_time,
            GRB.GREATER_EQUAL,
            0,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts_False",
        )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            1,
            task_1.start_time + task_1_remaining_time_expr - task_2.start_time,
            GRB.LESS_EQUAL,
            -1,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts_True",
        )

        # Add an indicator variable that is set to 1 if neither of the above
        # options can be evaluted to 1.
        if overlap_variable is None:
            overlap_variable = optimizer.addVar(
                vtype=GRB.BINARY, name=f"Overlap({task_1.name}, {task_2.name})"
            )
        optimizer.addConstr(
            task_1_starts_after_task_2_ends
            + task_1_ends_before_task_2_starts
            + overlap_variable
            == 1,
            name=f"{task_1.name}_overlap_{task_2.name}",
        )
        return overlap_variable

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
        workers: Mapping[int, Worker],
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

        if self._goal == "max_goodput":
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
                        task_remaining_time_expr = gp.LinExpr()
                        for worker_id in workers.keys():
                            for (
                                strategy
                            ) in task_variable.task.available_execution_strategies:
                                task_remaining_time_expr.add(
                                    task_variable.placed_on_worker_with_strategy(
                                        worker_id, strategy
                                    )
                                    * strategy.runtime.to(EventTime.Unit.US).time
                                )
                        optimizer.addGenConstrIndicator(
                            meets_deadline,
                            0,
                            task_variable.start_time + task_remaining_time_expr,
                            GRB.GREATER_EQUAL,
                            task_variable.task.deadline.to(EventTime.Unit.US).time + 1,
                            name=f"{task_variable.name}_meets_deadline_FALSE",
                        )
                        optimizer.addGenConstrIndicator(
                            meets_deadline,
                            1,
                            task_variable.start_time + task_remaining_time_expr,
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
        elif self._goal == "max_slack":
            optimization_expression = gp.QuadExpr()
            for (
                task_graph_name,
                task_graph_reward_variable,
            ) in task_graph_reward_variables.items():
                task_graph = workload.get_task_graph(task_graph_name)

                # Compute the reward for the deepest tasks released.
                for task_variable in tasks_to_variables.values():
                    if task_variable.task.task_graph == task_graph_name:
                        # Check if the task is a sink task.
                        # Either the task is a sink task (i.e., has no children), or
                        # none of its children are included in the currently released
                        # tasks during this invocation of the Scheduler.
                        is_sink_task = task_graph.is_sink_task(
                            task_variable.task
                        ) or not any(
                            child.unique_name in tasks_to_variables
                            for child in task_graph.get_children(task_variable.task)
                        )

                        if not is_sink_task:
                            continue

                        optimization_expression.add(
                            (
                                gp.quicksum(task_variable.placed_on_workers)
                                * (
                                    task_variable.task.deadline.to(
                                        EventTime.Unit.US
                                    ).time
                                    - task_variable.start_time
                                )
                            )
                        )
            optimizer.setObjective(optimization_expression, sense=GRB.MAXIMIZE)
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
