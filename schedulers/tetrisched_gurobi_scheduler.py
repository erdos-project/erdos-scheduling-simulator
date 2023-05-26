from itertools import islice
import multiprocessing
from operator import attrgetter
import sys
import time
from collections import defaultdict, deque
from copy import copy, deepcopy
from typing import Any, List, Mapping, Optional, Sequence, Set, Union
from uuid import UUID

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from schedulers import BaseScheduler
from schedulers.tetrisched_cplex_scheduler import BatchTask
from utils import EventTime
from workers import Worker, WorkerPools
from workload import Placement, Placements, Task, TaskState, Workload
from workload.profile import WorkProfile
from workload.strategy import ExecutionStrategies, ExecutionStrategy

UNPLACED_PENALTY = 99999


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
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
    """

    def __init__(
        self,
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        task: Union[Task, BatchTask],
        workers: Mapping[int, Worker],
        optimizer: gp.Model,
        enforce_deadlines: bool,
        retract_schedules: bool,
    ):
        self._task = task
        self._previously_placed = False
        self._is_placed_variable = None
        self._reward_variable = 0

        # Placement characteristics.
        # Set up a matrix of variables that signify the time, the worker and the
        # strategy with which the the task is placed.
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

        # Timing characteristics.
        if task.state == TaskState.RUNNING:
            # The task is already running, set the start time to the current
            # simulation time, since we use the remaining time to count the
            # time at which this Task will relinquish its resources.
            self._previously_placed = True
            placed_key = (
                self.__get_worker_index_from_previous_placement(task, workers),
                current_time.to(EventTime.Unit.US).time,
                task.current_placement.execution_strategy,
            )
            self._space_time_strategy_matrix[placed_key] = 1
            self._is_placed_variable = 1
            # We scale the reward to 2 so that it is in the same range as the rewards
            # being passed to SCHEDULED / RELEASED tasks (see below). We also multiply
            # the slack that was left for this Task.
            self._reward_variable = (
                (len(self._task.tasks) ** 2) * self._task.priority
                if isinstance(self._task, BatchTask)
                else 1
            ) * 2
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
                if len(compatible_strategies) != 0:
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

            if task.state == TaskState.SCHEDULED and not isinstance(task, BatchTask):
                # Maintain a warm-start cache that can be used to pass the starting
                # values to the optimizer.
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

            if task.state == TaskState.SCHEDULED and not retract_schedules:
                # If the task was previously scheduled, and we do not allow retractions,
                # we allow the start time to be fungible, but the task must be placed.
                optimizer.addConstr(
                    gp.quicksum(self._space_time_strategy_matrix.values()) == 1,
                    name=(
                        f"{task.unique_name}"
                        "_previously_scheduled_required_worker_placement"
                    ),
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
                    vtype=GRB.BINARY, name=f"{task.unique_name}_is_placed"
                )
                optimizer.addConstr(
                    self._is_placed_variable
                    == gp.quicksum(self._space_time_strategy_matrix.values()),
                    name=f"{task.unique_name}_is_placed_constraint",
                )

            # The reward for placing the task is the number of tasks in the batch if
            # the task is a `BatchTask` or 1 if the task is a `Task`.
            task_reward = (
                len(self._task.tasks) * (1 + len(self._task.tasks) * 0.01)
                if isinstance(self._task, BatchTask)
                else 1
            )

            # The placement reward skews the reward towards placing the task earlier.
            # We interpolate the time range to a range between 2 and 1 and use that to
            # skew the reward towards earlier placement.
            placement_rewards = dict(
                zip(
                    time_range,
                    np.interp(time_range, (min(time_range), max(time_range)), (2, 1)),
                )
            )

            # The slack reward skews the reward towards placing tasks with least slack
            # earlier. The slack reward is normalized to a range between 2 and 1.
            slack_reward = (
                self._task.priority if isinstance(self._task, BatchTask) else 1
            )

            # Set the reward variable according to the `task_reward`.
            self._reward_variable = optimizer.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=task_reward * 4,
                name=f"{task.unique_name}_reward",
            )

            # TODO: Ideally, this should use the strategy's batch size to
            # inform the reward as opposed to a fixed value of `task_reward`. However,
            # we only generate `BatchTask`s with a fixed strategy for now, and so this
            # formulation works.
            reward = []
            for (
                _,
                start_time,
                _,
            ), variable in self._space_time_strategy_matrix.items():
                reward.append(
                    task_reward
                    * placement_rewards[start_time]
                    * slack_reward
                    * variable
                )
            optimizer.addConstr(
                self._reward_variable == gp.quicksum(reward),
                name=f"{task.unique_name}_reward_constraint",
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

    @property
    def task(self) -> Union[Task, BatchTask]:
        """Returns the Task that this instance of TaskOptimizerVariables represents."""
        return self._task

    @property
    def previously_placed(self) -> bool:
        """Returns a Boolean indicating whether the task denoted by this instance
        has been previously placed or not."""
        return self._previously_placed

    @property
    def is_placed(self) -> Union[int, gp.Var]:
        """Check if the scheduler found a placement for the task."""
        return self._is_placed_variable

    def get_partition_variables(
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

    def get_placements(
        self,
        worker_index_to_worker: Mapping[int, Worker],
        worker_id_to_worker_pool: Mapping[UUID, UUID],
    ) -> Sequence[Placement]:
        """Constructs `Placement` objects for the scheduler to return the `Simulator`.

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
            return []
        for (
            worker_id,
            start_time,
            strategy,
        ), variable in self._space_time_strategy_matrix.items():
            if type(variable) != int and variable.X == 1:
                placement_worker = worker_index_to_worker[worker_id]
                placement_worker_pool_id = worker_id_to_worker_pool[placement_worker.id]
                if isinstance(self.task, BatchTask):
                    return [
                        Placement.create_task_placement(
                            task=task,
                            placement_time=EventTime(
                                start_time, unit=EventTime.Unit.US
                            ),
                            worker_pool_id=placement_worker_pool_id,
                            worker_id=placement_worker.id,
                            execution_strategy=strategy,
                        )
                        for task in self.task.tasks
                    ]
                else:
                    return [
                        Placement.create_task_placement(
                            task=self.task,
                            placement_time=EventTime(
                                start_time, unit=EventTime.Unit.US
                            ),
                            worker_pool_id=placement_worker_pool_id,
                            worker_id=placement_worker.id,
                            execution_strategy=strategy,
                        )
                    ]

        if isinstance(self.task, BatchTask):
            return [
                Placement.create_task_placement(
                    task=task,
                    placement_time=None,
                    worker_pool_id=None,
                    worker_id=None,
                    execution_strategy=None,
                )
                for task in self.task.tasks
            ]
        else:
            return [
                Placement.create_task_placement(
                    task=self.task,
                    placement_time=None,
                    worker_pool_id=None,
                    worker_id=None,
                    execution_strategy=None,
                )
            ]

    @property
    def reward(self) -> Optional[Union[int, gp.Var]]:
        """Returns the integer variable that denotes the reward obtained from the
        placement of this task."""
        return self._reward_variable


class TetriSchedGurobiScheduler(BaseScheduler):
    """Implements an TetriSched formulation of the scheduling problem for the Simulator
    using Gurobi.

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
        batching (`bool`) : If `True`, the scheduler will batch tasks together from the
            same `WorkProfile` if possible.
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
        retract_schedules: bool = False,
        goal: str = "max_goodput",
        batching: bool = False,
        time_limit: EventTime = EventTime(20, unit=EventTime.Unit.S),
        time_discretization: EventTime = EventTime(1, unit=EventTime.Unit.US),
        plan_ahead: EventTime = EventTime(-1, EventTime.Unit.US),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(TetriSchedGurobiScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=False,
            _flags=_flags,
        )
        self._goal = goal
        self._batching = batching
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

        # Run admission control. If `enforce_deadlines` is requested, we will drop the
        # tasks that are past its deadline.
        if self.enforce_deadlines:
            tasks_to_remove: List[Task] = []
            for task in tasks_to_be_scheduled:
                if (
                    task.deadline
                    < sim_time
                    + task.available_execution_strategies.get_fastest_strategy().runtime
                ):
                    tasks_to_remove.append(task)

            for task in tasks_to_remove:
                self._logger.debug(
                    "[%s] Requesting cancellation of %s since its "
                    "deadline is %s and the fastest strategy takes %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    task,
                    task.deadline.to(EventTime.Unit.US),
                    task.available_execution_strategies.get_fastest_strategy().runtime,
                )
                placements.append(Placement.create_task_cancellation(task))
                tasks_to_be_scheduled.remove(task)

        if len(tasks_to_be_scheduled) != 0:
            optimizer = self._initialize_optimizer(sim_time)
            tasks_to_variables = self._add_variables(
                sim_time,
                optimizer,
                tasks_to_be_scheduled + previously_placed_tasks,
                workers,
            )
            self._add_resource_constraints(
                sim_time, optimizer, tasks_to_variables, workers
            )
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
                # Keep a mapping of the current placements found for each `Task`, and
                # return the actual `Placement` objects once all the batches have been
                # looked through.
                task_placement_map: Mapping[Task, Placement] = {}
                for task_variable in tasks_to_variables.values():
                    # If the task was previously placed, we don't need to return new
                    # placements.
                    if task_variable.previously_placed:
                        continue

                    task_placements = task_variable.get_placements(
                        workers, worker_to_worker_pool
                    )
                    self._logger.debug(
                        "[%s] Received placements for [%s] with the "
                        "batching profile %s.",
                        sim_time.to(EventTime.Unit.US).time,
                        ", ".join(p.task.unique_name for p in task_placements),
                        task_variable.task.unique_name,
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
                        placements.append(placement)
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
        plan_ahead = self._plan_ahead
        if plan_ahead == EventTime(-1, EventTime.Unit.US):
            for task in tasks_to_be_scheduled:
                if task.deadline > plan_ahead:
                    plan_ahead = task.deadline
        if self._batching:
            # If batching is enabled, find the tasks that are to be batched together.
            profile_to_tasks: Mapping[WorkProfile, Set[Task]] = defaultdict(set)
            for task in tasks_to_be_scheduled:
                profile_to_tasks[task.profile].add(task)

            for profile, tasks in profile_to_tasks.items():
                tasks_to_variables.update(
                    self._create_batch_task_variables(
                        sim_time=sim_time,
                        plan_ahead=plan_ahead,
                        optimizer=optimizer,
                        profile=profile,
                        tasks=tasks,
                        workers=workers,
                    )
                )
        else:
            for task in tasks_to_be_scheduled:
                tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                    sim_time,
                    plan_ahead,
                    self._time_discretization,
                    task,
                    workers,
                    optimizer,
                    self.enforce_deadlines,
                    self.retract_schedules,
                )
        return tasks_to_variables

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
        plan_ahead = self._plan_ahead
        if plan_ahead == EventTime(-1, EventTime.Unit.US):
            for task_variable in tasks_to_variables.values():
                if task_variable.task.deadline > plan_ahead:
                    plan_ahead = task_variable.task.deadline

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
                    partition_variables = task_variable.get_partition_variables(
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
                    resource_constraint_terms = []
                    for strategy, task_overlap_variables in overlap_variables.items():
                        task_request_for_resource = (
                            strategy.resources.get_total_quantity(resource)
                        )
                        if task_request_for_resource == 0:
                            # If the task does not require this resource, we skip adding
                            # any of the terms into this resource constraint expression.
                            continue
                        for task_overlap_variable in task_overlap_variables:
                            resource_constraint_terms.append(
                                task_request_for_resource * task_overlap_variable
                            )
                    if quantity == 0 or len(resource_constraint_terms) == 0:
                        # If either the Worker doesn't have enough space to accomodate
                        # this task, or no task wants to occupy this resource at the
                        # particular time, we skip the addition of this constraint.
                        continue

                    optimizer.addConstr(
                        gp.quicksum(resource_constraint_terms) <= quantity,
                        name=(
                            f"{resource.name}_utilization_Worker"
                            f"_{worker_index}_at_Time_{t}"
                        ),
                    )

    def _create_batch_task_variables(
        self,
        sim_time: EventTime,
        plan_ahead: EventTime,
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
                    priority=sum(
                        [
                            task.deadline - sim_time - task.remaining_time
                            for task in tasks_for_this_strategy
                        ],
                        start=EventTime.zero(),
                    )
                    .to(EventTime.Unit.US)
                    .time,
                )
            )
            batch_counter += 1

        # Create `BatchTask`s for all the unscheduled tasks.
        unscheduled_tasks = deque(sorted(unscheduled_tasks, key=attrgetter("deadline")))
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

                # Find the tasks that need to fit into this batch, and create
                # a new `BatchTask`.
                tasks_for_this_strategy = list(
                    islice(unscheduled_tasks, 0, strategy.batch_size)
                )
                priority_for_this_strategy = (
                    sum(
                        [
                            task.deadline - sim_time - strategy.runtime
                            for task in tasks_for_this_strategy
                        ],
                        start=EventTime.zero(),
                    )
                    .to(EventTime.Unit.US)
                    .time
                )
                self._logger.debug(
                    "[%s] Creating the batching strategy %s with tasks: [%s], "
                    "and priority %d.",
                    sim_time.to(EventTime.Unit.US).time,
                    f"{profile.name}_{batch_counter}",
                    ", ".join([t.unique_name for t in tasks_for_this_strategy]),
                    priority_for_this_strategy,
                )
                batch_task = BatchTask(
                    name=f"{profile.name}_{batch_counter}",
                    tasks=tasks_for_this_strategy,
                    strategy=strategy,
                    priority=priority_for_this_strategy,
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

        # Now that we have all the `BatchTask`s, we first renormalize their priorities
        # and then construct a `BatchTaskVariable` for each of them.
        batch_priorities = [task.priority for task in batch_tasks]
        for batch_task, new_priority in zip(
            batch_tasks,
            np.interp(
                batch_priorities, (min(batch_priorities), max(batch_priorities)), (2, 1)
            ),
        ):
            batch_task.priority = new_priority

        # Create the `BatchTaskVariable`s for each of the `BatchTask`s.
        batch_task_variables: Mapping[str, TaskOptimizerVariables] = {}
        for batch_task in batch_tasks:
            batch_task_variables[batch_task.unique_name] = TaskOptimizerVariables(
                optimizer=optimizer,
                task=batch_task,
                workers=workers,
                current_time=sim_time,
                plan_ahead=plan_ahead,
                time_discretization=self._time_discretization,
                enforce_deadlines=self.enforce_deadlines,
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
            optimizer.addConstr(
                gp.quicksum(
                    [
                        batch_task_variable.is_placed
                        for batch_task_variable in map(
                            lambda task: batch_task_variables[task.unique_name],
                            batch_tasks,
                        )
                    ]
                )
                <= 1,
                name=f"{task.unique_name}_unique_batch_placement",
            )

        return batch_task_variables

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        if self._goal == "max_goodput":
            # Define reward variables for each of the tasks, that is a sum of their
            # space-time matrices. Maximizing the sum of these rewards is the goal of
            # the scheduler.
            task_not_placed_penalty = []
            if self._batching:
                tasks_to_batch_tasks: Mapping[
                    Task, Set[TaskOptimizerVariables]
                ] = defaultdict(set)
                for batch_task_variable in tasks_to_variables.values():
                    for task in batch_task_variable.task.tasks:
                        tasks_to_batch_tasks[task].add(batch_task_variable)

                for task, batch_task_variables in tasks_to_batch_tasks.items():
                    task_not_placed_variable = optimizer.addVar(
                        vtype=GRB.INTEGER,
                        lb=-1,
                        ub=0,
                        name=f"{task.unique_name}_not_placed",
                    )
                    optimizer.addConstr(
                        task_not_placed_variable
                        == (
                            gp.quicksum(
                                [
                                    batch_task_variable.is_placed
                                    for batch_task_variable in batch_task_variables
                                ]
                            )
                            - 1
                        )
                    )
                    task_not_placed_penalty.append(
                        task_not_placed_variable * UNPLACED_PENALTY
                    )
            optimizer.setObjective(
                gp.quicksum(
                    [
                        task_variable.reward
                        for task_variable in tasks_to_variables.values()
                    ]
                    + task_not_placed_penalty
                ),
                sense=GRB.MAXIMIZE,
            )
        else:
            raise RuntimeError(
                f"The goal {self._goal} is not supported yet by "
                f"TetriSchedCPLEXScheduler."
            )

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
