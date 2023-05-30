import multiprocessing
import random
import sys
import time
from collections import defaultdict, deque
from copy import copy, deepcopy
from itertools import combinations, islice
from operator import attrgetter
from typing import List, Mapping, Optional, Sequence, Set, Union
from uuid import UUID

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    BatchStrategy,
    BranchPredictionPolicy,
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
    WorkProfile,
)


class BatchTask(object):
    """A `BatchTask` is a virtual `Task` object that is used to represent a batch of
    tasks that are to be executed together using the given strategy.

    Args:
        name (`str`): The name of the batch task.
        tasks (`Sequence[Task]`): The list of tasks that are to be batched together.
        strategy (`ExecutionStrategy`): The strategy that is to be used to execute the
            batch of tasks.
    """

    def __init__(self, name: str, tasks: Sequence[Task], strategy: ExecutionStrategy):
        self._name = name
        self._tasks = tasks
        self._strategy = BatchStrategy(execution_strategy=strategy)
        self._id = UUID(int=random.getrandbits(128), version=4)
        self._hash = hash(self._id)

    @property
    def state(self) -> TaskState:
        if all(task.state == TaskState.RUNNING for task in self._tasks):
            return TaskState.RUNNING
        elif all(task.state == TaskState.SCHEDULED for task in self._tasks):
            return TaskState.SCHEDULED
        else:
            return TaskState.RELEASED

    @property
    def expected_start_time(self) -> EventTime:
        return self._tasks[0].expected_start_time

    @property
    def current_placement(self) -> Placement:
        return (self._tasks[0]).current_placement

    @property
    def unique_name(self) -> str:
        return self._name

    @property
    def release_time(self) -> EventTime:
        return max(task.release_time for task in self._tasks)

    @property
    def deadline(self) -> EventTime:
        return min(task.deadline for task in self._tasks)

    @property
    def available_execution_strategies(self) -> ExecutionStrategies:
        return ExecutionStrategies(strategies=[self._strategy])

    @property
    def tasks(self) -> Sequence[Task]:
        return self._tasks

    def __hash__(self) -> int:
        return self._hash


class TaskOptimizerVariables(object):
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
        task: Union[Task, BatchTask],
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
            if task.state == TaskState.SCHEDULED and not isinstance(task, BatchTask):
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

    def get_placements(
        self,
        worker_index_to_workers: Mapping[int, Worker],
        worker_id_to_worker_pool: Mapping[UUID, UUID],
    ) -> Sequence[Placement]:
        """Retrieves the details of the solution and constructs the `Placement`
        objects for the Scheduler to return to the Simulator.

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
        # If the task corresponding to these variables was previously placed, then
        # no new Placements will be returned.
        if self.previously_placed:
            return []

        # Find the start time of the task, and the Worker it was placed on along with
        # the strategy that is to be used to execute this Task.
        start_time = EventTime(int(self.start_time.X), EventTime.Unit.US)
        placement_worker_id = None
        placement_worker_pool_id = None
        placement_strategy = None
        for worker_id, worker in worker_index_to_workers.items():
            for strategy in self.task.available_execution_strategies:
                placement_variable = self.placed_on_worker_with_strategy(
                    worker_id, strategy
                )
                if isinstance(placement_variable, gp.Var) and placement_variable.X == 1:
                    placement_worker_id = worker.id
                    placement_worker_pool_id = worker_id_to_worker_pool[worker.id]
                    placement_strategy = strategy
                    break

        # Construct the Placement object for the Task.
        tasks_to_place = []
        if isinstance(self.task, BatchTask):
            tasks_to_place.extend(self.task.tasks)
        else:
            tasks_to_place.append(self.task)
        return [
            Placement.create_task_placement(
                task=task,
                placement_time=start_time
                if placement_worker_id and placement_strategy
                else None,
                worker_pool_id=placement_worker_pool_id,
                worker_id=placement_worker_id,
                execution_strategy=placement_strategy,
            )
            for task in tasks_to_place
        ]


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
        if len(tasks_to_be_scheduled) > 0:
            optimizer = self._initialize_optimizer(sim_time)
            tasks_to_variables = self._add_variables(
                sim_time,
                optimizer,
                workload,
                tasks_to_be_scheduled + previously_placed_tasks,
                workers,
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

                task_placement_map: Mapping[Task, Placement] = {}
                for task_variable in tasks_to_variables.values():
                    if task_variable.previously_placed:
                        continue

                    task_placements = task_variable.get_placements(
                        workers, worker_to_worker_pool
                    )
                    for placement in task_placements:
                        if (
                            placement.task not in task_placement_map
                            or not task_placement_map[placement.task].is_placed()
                        ):
                            self._logger.debug(
                                "[%s] Placing task %s as part of strategy %s "
                                "with the placement: %s.",
                                sim_time.to(EventTime.Unit.US).time,
                                placement.task.unique_name,
                                task_variable.task.unique_name,
                                str(placement),
                            )
                            task_placement_map[placement.task] = placement

                for task, placement in task_placement_map.items():
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
                            task.unique_name,
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

    def _create_batch_task_variables(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        profile: WorkProfile,
        tasks: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
        # Sanity check that all tasks are from the same profile.
        for task in tasks:
            if task.profile != profile:
                raise ValueError(
                    f"Task {task.unique_name} has profile {task.profile.name}, "
                    f"but was expected to have profile {profile.name}."
                )

        # Seperate the tasks into running or unscheduled.
        running_tasks: Mapping[ExecutionStrategy, List[Task]] = defaultdict(list)
        unscheduled_tasks: Sequence[Task] = []
        for task in tasks:
            if task.state == TaskState.RUNNING or (
                not self._retract_schedules and task.state == TaskState.SCHEDULED
            ):
                running_tasks[task.current_placement.execution_strategy].append(task)
            else:
                unscheduled_tasks.append(task)

        # Create `BatchTask`s for all the running tasks.
        batch_tasks: List[BatchTask] = []
        batch_counter = 1
        for execution_strategy, tasks_for_this_strategy in running_tasks.items():
            batch_tasks.append(
                BatchTask(
                    name=f"{profile.name}_{batch_counter}",
                    tasks=tasks_for_this_strategy,
                    strategy=execution_strategy,
                )
            )
            batch_counter += 1

        # Create `BatchTask`s for all the unscheduled tasks.
        # Tasks that are allowed to miss deadlines are assigned a deadline of infinity.
        unscheduled_tasks = deque(
            sorted(
                unscheduled_tasks,
                key=lambda t: t.deadline
                if t.task_graph not in self._allowed_to_miss_deadlines
                else float("inf"),
            )
        )
        tasks_to_batch_tasks: Mapping[Task, List[BatchTask]] = defaultdict(list)
        while len(unscheduled_tasks) > 0:
            # Get the strategies that satisfy the deadline for the first task.
            earliest_deadline_task = unscheduled_tasks[0]
            compatible_strategies = ExecutionStrategies()
            for strategy in sorted(
                earliest_deadline_task.available_execution_strategies,
                key=attrgetter("batch_size"),
                reverse=True,
            ):
                if sim_time + strategy.runtime <= earliest_deadline_task.deadline:
                    compatible_strategies.add_strategy(strategy)

            # Construct `BatchTask`s for strategies that can satisfy the batch size
            # requirements.
            for strategy in compatible_strategies:
                if len(unscheduled_tasks) < strategy.batch_size:
                    continue

                # Find the tasks that need to fit into this batch, and create a new
                # `BatchTask` for them.
                tasks_for_this_strategy = list(
                    islice(unscheduled_tasks, 0, strategy.batch_size)
                )
                self._logger.debug(
                    "[%s] Creating the batching strategy %s with tasks: [%s].",
                    sim_time.to(EventTime.Unit.US).time,
                    f"{profile.name}_{batch_counter}",
                    ", ".join([t.unique_name for t in tasks_for_this_strategy]),
                )
                batch_task = BatchTask(
                    name=f"{profile.name}_{batch_counter}",
                    tasks=tasks_for_this_strategy,
                    strategy=strategy,
                )

                # Add the `BatchTask` to the list of `BatchTask`s, and keep track of
                # which `BatchTask`s were associated with each schedulable `Task` so
                # we can allow only one of them to be scheduled.
                batch_tasks.append(batch_task)
                for task in tasks_for_this_strategy:
                    tasks_to_batch_tasks[task].append(batch_task)
                batch_counter += 1

            # Move on to the next task.
            unscheduled_tasks.popleft()

        # Create the `BatchTaskVariable`s for each of the `BatchTask`s.
        batch_task_variables: Mapping[str, TaskOptimizerVariables] = {}
        for batch_task in batch_tasks:
            batch_task_variables[batch_task.unique_name] = TaskOptimizerVariables(
                current_time=sim_time,
                task=batch_task,
                workers=workers,
                optimizer=optimizer,
                # Deadlines are not to be enforced if the tasks in this batch are all
                # allowed to miss their deadlines. Otherwise, we conservatively aim to
                # enforce deadlines.
                enforce_deadlines=False
                if all(
                    task.task_graph in self._allowed_to_miss_deadlines
                    for task in batch_task.tasks
                )
                else self.enforce_deadlines,
                retract_schedules=self.retract_schedules,
            )

        # Ensure that only one of the `BatchTask`s associated with each `Task` is
        # scheduled.
        for task, batch_tasks in tasks_to_batch_tasks.items():
            self._logger.debug(
                "[%s] Ensuring that only one of [%s] is placed for task %s.",
                sim_time.to(EventTime.Unit.US).time,
                ", ".join([batch_task.unique_name for batch_task in batch_tasks]),
                task.unique_name,
            )

            placement_variables = []
            for batch_task_variable in map(
                lambda bt: batch_task_variables[bt.unique_name], batch_tasks
            ):
                placement_variables.extend(batch_task_variable.placed_on_workers)
            optimizer.addConstr(
                gp.quicksum(placement_variables) <= 1,
                name=f"{task.unique_name}_unique_batch_placement",
            )

        return batch_task_variables

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

        if self._batching:
            # If batching is enabled, find the tasks that share a `WorkProfile` and
            # can be batched together.
            profile_to_tasks: Mapping[WorkProfile, Set[Task]] = defaultdict(set)
            for task in tasks_to_be_scheduled:
                profile_to_tasks[task.profile].add(task)

            for profile, tasks in profile_to_tasks.items():
                tasks_to_variables.update(
                    self._create_batch_task_variables(
                        sim_time, optimizer, profile, tasks, workers
                    )
                )
        else:
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

            parent_tasks = []
            parent_variables = []
            if isinstance(variable.task, BatchTask):
                # If the task is a `BatchTask`, we need to ensure the following:
                # 1. The BatchTask is placed only if all the parent tasks of each Task
                #    in the BatchTask are placed.
                # 2. The BatchTask is placed after a BatchTask containing the last
                #    parent of each Task in the BatchTask to finish has finished.
                for task in variable.task.tasks:
                    task_graph = workload.get_task_graph(task.task_graph)
                    parent_tasks.extend(task_graph.get_parents(task))

                parent_variables = [
                    variable
                    for variable in tasks_to_variables.values()
                    if isinstance(variable.task, BatchTask)
                    and any(
                        parent_task in variable.task.tasks
                        for parent_task in parent_tasks
                    )
                ]
                for variable in tasks_to_variables.values():
                    if isinstance(variable.task, BatchTask) and any(
                        parent_task in variable.task.tasks
                        for parent_task in parent_tasks
                    ):
                        parent_variables.append(variable)
            else:
                # If the task is a `Task`, we need to ensure the following:
                # 1. The Task is only placed if all of its parents are placed.
                # 2. The Task is started after the last parent has finished.
                task_graph = workload.get_task_graph(variable.task.task_graph)
                parent_tasks = task_graph.get_parents(variable.task)
                parent_variables = [
                    tasks_to_variables[parent.unique_name]
                    for parent in parent_tasks
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
                    len(parent_tasks) - 1,
                    name=f"{task_name}_parents_placed_False",
                )
                optimizer.addGenConstrIndicator(
                    all_parents_placed,
                    1,
                    gp.quicksum(parent_placements),
                    GRB.EQUAL,
                    len(parent_tasks),
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
                task_graph_to_tasks: Mapping[str, List[Task]] = defaultdict(list)
                if isinstance(task_1_variable.task, BatchTask) and isinstance(
                    task_2_variable.task, BatchTask
                ):
                    for task in task_1_variable.task.tasks:
                        task_graph_to_tasks[task.task_graph].append(task)
                    for task in task_2_variable.task.tasks:
                        task_graph_to_tasks[task.task_graph].append(task)
                else:
                    task_graph_to_tasks[task_1_variable.task.task_graph].append(
                        task_1_variable.task
                    )
                    task_graph_to_tasks[task_2_variable.task.task_graph].append(
                        task_2_variable.task
                    )

                overlaps = True
                for task_graph_name, tasks in task_graph_to_tasks.items():
                    task_graph = workload.get_task_graph(task_graph_name)
                    if any(
                        task_graph.are_dependent(task_1, task_2)
                        for task_1, task_2 in combinations(tasks, r=2)
                    ):
                        # If any of the tasks are dependent on each other,
                        # these pair of task variables can never overlap.
                        optimizer.addConstr(
                            task_pair_overlap_variable == 0,
                            name=f"{task_1_name}_no_overlap_{task_2_name}_dependent",
                        )
                        overlaps = False
                        break

                if overlaps:
                    # If the tasks are not dependent on each other, they may overlap.
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
            tasks = (
                task_variable.task.tasks
                if isinstance(task_variable.task, BatchTask)
                else [task_variable.task]
            )
            for task in tasks:
                task_graph = workload.get_task_graph(task.task_graph)
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

                # Find the tasks for this TaskGraph that will lead to a reward.
                # If the `release_taskgraphs` option was set, the sink tasks of the
                # TaskGraph provide a reward. Otherwise, the reward is provided by
                # the deepest tasks in the TaskGraph that were schedulable.
                reward_tasks: List[Task] = []
                for task_variable in tasks_to_variables.values():
                    # Find the tasks in this variable that belong to this TaskGraph.
                    tasks_in_this_variable = []
                    if (
                        isinstance(task_variable.task, Task)
                        and task_variable.task.task_graph == task_graph_name
                    ):
                        tasks_in_this_variable.append(task_variable.task)
                    elif isinstance(task_variable.task, BatchTask):
                        for task in task_variable.task.tasks:
                            if task.task_graph == task_graph_name:
                                tasks_in_this_variable.append(task)

                    # From all the tasks in this variable, find the ones that can
                    # potentially provide a reward.
                    for task in tasks_in_this_variable:
                        is_reward_task = True
                        if self.release_taskgraphs:
                            is_reward_task = task_graph.is_sink_task(task)
                        else:
                            for child_task in task_graph.get_children(task):
                                for task_variable in tasks_to_variables.values():
                                    if (
                                        isinstance(task_variable.task, Task)
                                        and task_variable.task == child_task
                                    ) or (
                                        isinstance(task_variable.task, BatchTask)
                                        and child_task in task_variable.task.tasks
                                    ):
                                        is_reward_task = False
                                        break

                        if is_reward_task:
                            reward_tasks.append(task)

                # For all the reward tasks, find the actual reward that is achieved
                # by placement of the Task itself or across all its BatchTasks.
                task_reward_variables = []
                for reward_task in reward_tasks:
                    # Find the variables that contribute to the reward of this task.
                    variables_for_reward_task = []
                    for task_variable in tasks_to_variables.values():
                        if (
                            isinstance(task_variable.task, Task)
                            and task_variable.task == reward_task
                        ) or (
                            isinstance(task_variable.task, BatchTask)
                            and reward_task in task_variable.task.tasks
                        ):
                            variables_for_reward_task.extend(
                                task_variable.placed_on_workers
                            )

                    # For all the variables that this Task can be placed as a part of,
                    # construct the reward variable as the placement reward across all
                    # the variables.
                    task_reward_variable = optimizer.addVar(
                        vtype=GRB.BINARY, name=f"{reward_task.unique_name}_reward"
                    )
                    optimizer.addConstr(
                        task_reward_variable == gp.quicksum(variables_for_reward_task),
                        name=f"{reward_task.unique_name}_reward_constraint",
                    )
                    task_reward_variables.append(task_reward_variable)

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
