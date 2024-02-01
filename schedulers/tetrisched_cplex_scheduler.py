import multiprocessing
import random
import sys
import time
from collections import defaultdict, deque
from copy import copy, deepcopy
from itertools import islice
from operator import attrgetter
from typing import Any, List, Mapping, Optional, Sequence, Set, TextIO, Tuple, Union
from uuid import UUID

import absl  # noqa: F401
import docplex.mp.dvar as cpx_var
import docplex.mp.model as cpx
import numpy as np
from cplex.callbacks import MIPInfoCallback
from docplex.mp.environment import Environment
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    BatchStrategy,
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Placements,
    Task,
    TaskState,
    Workload,
    WorkProfile,
)

UNPLACED_PENALTY = 99999


class TerminationCheckCallback(MIPInfoCallback):
    def __init__(self, environment: Environment) -> None:
        MIPInfoCallback.__init__(self, environment)
        self._sim_time = None
        self._current_gap = float("inf")
        self._last_gap_update = time.time()
        self._solution_found = False
        self._gap_time_limit = None
        self._logger = None

    def __call__(self) -> None:
        # If the gap has changed, update the gap and the time at which it was changed.
        if (
            abs(self.get_MIP_relative_gap() - self._current_gap)
            > sys.float_info.epsilon
        ):
            self._logger.debug(
                "[%s] The gap between the incumbent (%f) and "
                "the best bound (%f) was updated from %f to %f.",
                self._sim_time.to(EventTime.Unit.US).time,
                self.get_incumbent_objective_value(),
                self.get_best_objective_value(),
                self._current_gap,
                self.get_MIP_relative_gap(),
            )
            self._solution_found = True
            self._current_gap = self.get_MIP_relative_gap()
            self._last_gap_update = time.time()

        # If the time since the last gap update is greater than the gap update, then
        # we terminate the solution search.
        solver_time = time.time()
        if (
            solver_time - self._last_gap_update > self._gap_time_limit
        ) and self._solution_found:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't changed in %f"
                "seconds, and there is a valid solution available. Terminating.",
                self._sim_time.to(EventTime.Unit.US).time,
                solver_time - self._last_gap_update,
            )
            self.abort()
        elif solver_time - self._last_gap_update > self._gap_time_limit:
            self._logger.debug(
                "[%s] The gap between the incumbent and best bound hasn't changed in %f"
                "seconds, and no valid solution has yet been found. Terminating.",
                self._sim_time.to(EventTime.Unit.US).time,
                solver_time - self._last_gap_update,
            )
            self.abort()


class BatchTask(object):
    """A `BatchTask` is a virtual `Task` object that is used to represent a batch of
    tasks that are to be executed together using the given strategy.

    Args:
        name (`str`): The name of the batch task.
        tasks (`Sequence[Task]`): The list of tasks that are to be batched together.
        strategy (`ExecutionStrategy`): The strategy that is to be used to execute the
            batch of tasks.
        priority (`float`): The priority of the batch task.  Usually defined as the
            normalized slack attributed to this BatchTask. The slack is defined as the
            minimum of the slack across all the tasks in this batch.
    """

    def __init__(
        self,
        name: str,
        tasks: Sequence[Task],
        strategy: ExecutionStrategy,
        priority: float,
    ) -> None:
        self._name = name
        self._tasks = tasks
        self._strategy = BatchStrategy(execution_strategy=strategy)
        self._priority = priority
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
    def priority(self) -> float:
        return self._priority

    @priority.setter
    def priority(self, priority: float) -> None:
        self._priority = priority

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
        optimizer (`gp.Model`): The instance of the Gurobi model to which the
            variables and constraints must be added.
        task (`Task`): The Task instance for which the variables are generated.
        workers (`Mapping[int, Worker]`): A mapping of the unique index of the
            Worker to its instance.
        current_time (`EventTime`): The time at which the scheduler was invoked.
            This is used to set a lower bound on the placement time of the tasks.
        enforce_deadlines (`bool`): If `True`, the scheduler tries to enforce
            deadline constraints on the tasks.
    """

    def __init__(
        self,
        optimizer: cpx.Model,
        task: Union[Task, BatchTask],
        workers: Mapping[int, Worker],
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        enforce_deadlines: bool = True,
        retract_schedules: bool = False,
    ) -> None:
        self._task = task
        self._previously_placed = False
        self._is_placed_variable = None
        self._reward_variable = None

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

        # Warm-Start Cache.
        # This is used to cache the MIP_START solution for the available variables, if
        # the task was previously placed. We maintain the cache so we can pass them
        # together to the model.
        self._warm_start_cache = {}

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
                    ] = optimizer.binary_var(
                        name=(
                            f"{task.unique_name}_placed_at_Worker"
                            f"_{worker_id}_on_Time_{start_time}"
                            f"_with_strategy_{strategy.id}"
                        )
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
                    if not isinstance(binary_variable, cpx_var.Var):
                        continue
                    if space_time_index == placed_key:
                        self._warm_start_cache[binary_variable] = 1
                    else:
                        self._warm_start_cache[binary_variable] = 0

            if task.state == TaskState.SCHEDULED and not retract_schedules:
                # If the task was previously scheduled, and we do not allow retractions,
                # we allow the start time to be fungible, but the task must be placed.
                optimizer.add_constraint(
                    ct=optimizer.sum(self._space_time_strategy_matrix.values()) == 1,
                    ctname=f"{task.unique_name}_previously_scheduled_"
                    f"required_worker_placement",
                )
                self._is_placed_variable = 1
            else:
                # If either the task was not previously placed, or we are allowing
                # retractions, then the task can be placed or left unplaced.
                optimizer.add_constraint(
                    ct=optimizer.sum(self._space_time_strategy_matrix.values()) <= 1,
                    ctname=f"{task.unique_name}_consistent_worker_placement",
                )
                self._is_placed_variable = optimizer.binary_var(
                    name=f"{task.unique_name}_is_placed"
                )
                optimizer.add_constraint(
                    ct=self._is_placed_variable
                    == optimizer.sum(self._space_time_strategy_matrix.values()),
                    ctname=f"{task.unique_name}_is_placed_constraint",
                )

            # The reward for placing the task is the number of tasks in the batch if
            # the task is a `BatchTask` or 1 if the task is a `Task`.
            task_reward = (
                len(self._task.tasks) * (1 + (len(self._task.tasks) * 0.01))
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
            # slack_reward = (
            #     self._task.priority if isinstance(self._task, BatchTask) else 1
            # )

            # Set the reward variable according to the `task_reward`.
            self._reward_variable = optimizer.continuous_var(
                lb=0,
                ub=task_reward * 4,
                name=f"{task.unique_name}_reward",
            )

            # TODO (Sukrit): Ideally, this should use the strategy's batch size to
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
                    # * slack_reward
                    * variable
                )
            optimizer.add_constraint(
                ct=(self._reward_variable == optimizer.sum(reward)),
                ctname=f"{task.unique_name}_reward_constraint",
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
        solution: SolveSolution,
        worker_index_to_worker: Mapping[int, Worker],
        worker_id_to_worker_pool: Mapping[UUID, UUID],
    ) -> Sequence[Placement]:
        """Retrieves the details of the solution from the given `SolveSolution` object,
        and constructs the `Placement` objects for the Scheduler to return to the
        Simulator.

        Args:
            solution (`SolveSolution`): The solution computed by the optimizer.
            worker_index_to_worker (`Mapping[int, Worker]`): A mapping from the index
                that the Worker was assigned for this scheduling run to a reference to
                the `Worker` itself.
            worker_id_to_worker_pool (`Mapping[UUID, UUID]`): A mapping from the ID of
                the `Worker` to the ID of the `WorkerPool` which it is a part of.

        Returns:
            A sequence of `Placement` objects depicting the time when the Task(s) are
            to be started, and the Worker where the Task(s) are to be executed.
        """
        if not solution or self.previously_placed:
            # If there was no solution, or the task was previously placed, then we
            # don't have to return a `Placement` object to the Simulator.
            return []

        for (
            worker_id,
            start_time,
            strategy,
        ), variable in self._space_time_strategy_matrix.items():
            if type(variable) != int and round(solution.get_value(variable)) == 1:
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

    def get_partition_variable(
        self, time: int, worker_index: int
    ) -> Mapping[ExecutionStrategy, Sequence[Union[cpx_var.Var, int]]]:
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
                and (type(variable) == cpx_var.Var or variable == 1)
            ):
                partition_variables[strategy].append(variable)
        return partition_variables

    @property
    def task(self) -> Task:
        return self._task

    @property
    def previously_placed(self) -> bool:
        """Returns a boolean variable indicating if the Task that this instance
        represents was previously placed (`True`), or being considered for scheduling
        during this execution (`False`)."""
        return self._previously_placed

    @property
    def warm_start_cache(self) -> Mapping[cpx.BinaryVarType, int]:
        """Returns the (possibly empty) MIP Start values for the variables in
        this instance."""
        return self._warm_start_cache

    @property
    def space_time_matrix(
        self,
    ) -> Mapping[Tuple[int, int, ExecutionStrategy], cpx.BinaryVarType]:
        """Returns a mapping from the (Worker Index, Time, ExecutionStrategy) to the
        binary variable specifying if the task was placed at that time or not."""
        return self._space_time_strategy_matrix

    @property
    def is_placed(self) -> Optional[Union[int, cpx.BinaryVarType]]:
        """Returns the binary variable that specifies if the task was placed or not."""
        return self._is_placed_variable

    @property
    def reward(self) -> Optional[Union[int, cpx.ContinuousVarType]]:
        """Returns the integer variable that denotes the reward obtained from the
        placement of this task."""
        return self._reward_variable


class TetriSchedCPLEXScheduler(BaseScheduler):
    """Implements a space-seperated ILP formulation of the scheduling problem for the
    Simulator.

    The algorithm closely follows the TetriSched methodology of scheduling, and uses
    the same solver. As a result, the scheduler cannot work with DAGs and can only
    schedule individually released tasks. However, this implementation seeks to provide
    further improvements over the TetriSched strategy especially pertaining to batching
    requests whenever possible.

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
    ) -> None:
        if preemptive:
            raise ValueError(
                "The TetriSched scheduler does not allow tasks to be preempted."
            )
        super(TetriSchedCPLEXScheduler, self).__init__(
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
        self._gap_time_limit = time_limit.to(EventTime.Unit.S).time
        self._time_discretization = time_discretization
        self._plan_ahead = plan_ahead
        self._log_to_file = log_to_file
        self._log_times = set(map(int, _flags.scheduler_log_times)) if _flags else set()

    def _initialize_optimizer(
        self, current_time: EventTime
    ) -> Tuple[cpx.Model, Optional[TextIO]]:
        """Initializes the Optimizer and sets the required parameters.

        Args:
            current_time (`EventTime`): The time at which the model was supposed to be
                invoked.

        Returns:
            A tuple containing the optimizer as the first argument, and a handle to
            the log file if requested by self._log_to_file.
        """
        optimizer = cpx.Model(f"TetriSched_{current_time.to(EventTime.Unit.US).time}")

        # Set the number of threads for this machine.
        optimizer.context.cplex_parameters.threads = multiprocessing.cpu_count()

        # Enable very verbose logging of the solution, if requested.
        output_file = None
        if (
            self._log_to_file
            or current_time.to(EventTime.Unit.US).time in self._log_times
        ):
            output_file = open(
                f"./tetrisched_cplex_{current_time.to(EventTime.Unit.US).time}.log", "w"
            )
            optimizer.context.cplex_parameters.mip.display = 5
            optimizer.set_log_output(output_file)

        return (optimizer, output_file)

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: "WorkerPools"
    ) -> Placements:
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
            f"{[f'{t.unique_name} ({t.deadline})' for t in tasks_to_be_scheduled]}."
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
            # Set up the problem variables, constraints and add the objectives.
            # If requested, log the MIP into a LP file.
            (optimizer, log_file) = self._initialize_optimizer(sim_time)
            tasks_to_variables = self._add_variables(
                sim_time=sim_time,
                optimizer=optimizer,
                tasks_to_be_scheduled=tasks_to_be_scheduled + previously_placed_tasks,
                workers=workers,
            )
            self._add_resource_constraints(
                sim_time=sim_time,
                optimizer=optimizer,
                tasks_to_variables=tasks_to_variables,
                workers=workers,
            )
            self._add_objective(
                optimizer=optimizer, tasks_to_variables=tasks_to_variables
            )

            # Add the termination check callback, and set the initial gap and the time
            # at which the gap was last updated.
            termination_check_callback = optimizer.register_callback(
                TerminationCheckCallback
            )
            termination_check_callback._sim_time = sim_time
            termination_check_callback._logger = self._logger
            termination_check_callback._gap_time_limit = self._gap_time_limit

            # Add the Warm Start caches to the problem.
            warm_start = {}
            for task_variable in tasks_to_variables.values():
                warm_start |= task_variable.warm_start_cache

            if len(warm_start) > 0:
                optimizer.add_mip_start(
                    mip_start_sol=SolveSolution(
                        model=optimizer,
                        var_value_map=warm_start,
                        name="tetrisched_warm_start",
                    )
                )
            if (
                self._log_to_file
                or sim_time.to(EventTime.Unit.US).time in self._log_times
            ):
                with open(
                    f"./tetrisched_cplex_{sim_time.to(EventTime.Unit.US).time}.lp", "w"
                ) as lp_out:
                    lp_out.write(optimizer.export_as_lp_string())

            # Solve the problem.
            solver_start_time = time.time()
            solution: SolveSolution = optimizer.solve()
            solver_end_time = time.time()
            solver_runtime = EventTime(
                int((solver_end_time - solver_start_time) * 1e6), EventTime.Unit.US
            )
            if solution:
                # A valid solution was found. Construct the Placement objects.
                self._logger.debug(
                    "[%s] The scheduler returned the objective value %s in %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    solution.objective_value,
                    solver_runtime,
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
                        solution, workers, worker_to_worker_pool
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

                # Write the solution to the SOL file, if requested.
                if (
                    self._log_to_file
                    or sim_time.to(EventTime.Unit.US).time in self._log_times
                ):
                    file_name = (
                        f"tetrisched_cplex_{sim_time.to(EventTime.Unit.US).time}.sol"
                    )
                    self._logger.debug(
                        f"[{sim_time.to(EventTime.Unit.US).time}] The solution for "
                        f"the problem is being written to {file_name}."
                    )
                    solution.export_as_sol(
                        path=".",
                        basename=file_name,
                    )
            else:
                # The solution was infeasible. Retrieve the details.
                solution_status: SolveDetails = optimizer.solve_details
                self._logger.info(
                    "[%s] Failed to place any task because the solver returned %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    solution_status.status,
                )
                # Cancel the tasks that were to be scheduled, but keep prior placements
                # for previously scheduled tasks.
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

            # Perform cleanup.
            if log_file is not None:
                # If a log file was requested, close it now.
                log_file.close()
            optimizer.end()

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
        plan_ahead: EventTime,
        optimizer: cpx.Model,
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
            optimizer.add_constraint(
                ct=optimizer.sum(
                    [
                        batch_task_variable.is_placed
                        for batch_task_variable in map(
                            lambda task: batch_task_variables[task.unique_name],
                            batch_tasks,
                        )
                    ]
                )
                <= 1,
                ctname=f"{task.unique_name}_unique_batch_placement",
            )

        return batch_task_variables

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: cpx.Model,
        tasks_to_be_scheduled: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
        """Generates the variables for the optimization problem.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            optimizer (`cpx.Model`): The instance of the Gurobi model to which the
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
            # If batching is disabled, create a variable for each task.
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

    def _add_resource_constraints(
        self,
        sim_time: EventTime,
        optimizer: cpx.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workers: Mapping[int, Worker],
    ):
        """Generates the linear constraints that ensure that none of the `Resource`
        types available on any `Worker` are oversubscribed in any time discretization.

        Args:
            sim_time (`EventTime`): The time at which the scheduler was invoked.
            optimizer (`cpx.Model`): The optimization model to which the constraints
                are to be added.
            tasks_to_variables (`Mapping[str, TaskOptimizerVariables]`): The mapping
                from the task names to the `TaskOptimizerVariables` instance that
                contains its representation of the space-time matrix.
            workers (`Mapping[int, Worker]`): A Mapping from the index assigned to each
                Worker in this scheduling run to a reference to the `Worker` itself.
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
                    optimizer.add_constraint(
                        ct=optimizer.sum(resource_constraint_terms) <= quantity,
                        ctname=(
                            f"{resource.name}_utilization_Worker"
                            f"_{worker_index}_at_Time_{t}"
                        ),
                    )

    def _add_objective(
        self,
        optimizer: cpx.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
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
            task_not_placed_penalty = []
            if self._batching:
                tasks_to_batch_tasks: Mapping[Task, Set[TaskOptimizerVariables]] = (
                    defaultdict(set)
                )
                for batch_task_variable in tasks_to_variables.values():
                    for task in batch_task_variable.task.tasks:
                        tasks_to_batch_tasks[task].add(batch_task_variable)

                for task, batch_task_variables in tasks_to_batch_tasks.items():
                    task_not_placed_variable = optimizer.integer_var(
                        lb=-1, ub=0, name=f"{task.unique_name}_not_placed"
                    )
                    optimizer.add_constraint(
                        task_not_placed_variable
                        == (
                            optimizer.sum(
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

            optimizer.maximize(
                optimizer.sum(
                    [
                        task_variable.reward
                        for task_variable in tasks_to_variables.values()
                    ]
                    + task_not_placed_penalty
                )
            )
        else:
            raise RuntimeError(
                f"The goal {self._goal} is not supported yet by "
                f"TetriSchedCPLEXScheduler."
            )
