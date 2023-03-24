import multiprocessing
import time
from copy import copy, deepcopy
from typing import Mapping, Optional, Sequence, TextIO, Tuple, Union
from uuid import UUID

import absl  # noqa: F401
import docplex.mp.dvar as cpx_var
import docplex.mp.model as cpx
from docplex.mp.sdetails import SolveDetails
from docplex.mp.solution import SolveSolution

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import Placement, Placements, Task, TaskState, Workload


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
        task: Task,
        workers: Mapping[int, Worker],
        current_time: EventTime,
        plan_ahead: EventTime,
        time_discretization: EventTime,
        enforce_deadlines: bool = True,
        retract_schedules: bool = False,
    ) -> None:
        self._task = task
        self._previously_placed = False

        # Placement characteristics.
        # Set up a matrix of variables that signify the time and the worker where
        # the task is placed.
        self._space_time_matrix = {
            (worker_id, t): 0
            for worker_id in workers.keys()
            for t in range(
                current_time.to(EventTime.Unit.US).time,
                current_time.to(EventTime.Unit.US).time
                + plan_ahead.to(EventTime.Unit.US).time
                + 1,
                time_discretization.to(EventTime.Unit.US).time,
            )
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
            )
            self._space_time_matrix[placed_key] = 1
        else:
            # Initialize all the possible placement opportunities for this task into the
            # space-time matrix. The worker has to be able to accomodate the task, and
            # the timing constraints as requested by the Simulator have to be met.
            schedulable_workers = set()
            for worker_id, worker in workers.items():
                cleared_worker = deepcopy(worker)
                if cleared_worker.can_accomodate_task(task):
                    schedulable_workers.add(worker_id)

            for worker_id, start_time in self._space_time_matrix.keys():
                if start_time < task.release_time.to(EventTime.Unit.US).time:
                    # The time is before the task gets released, the task cannot be
                    # placed here.
                    self._space_time_matrix[(worker_id, start_time)] = 0
                elif (
                    enforce_deadlines
                    and start_time + task.remaining_time.to(EventTime.Unit.US).time
                    > task.deadline.to(EventTime.Unit.US).time
                ):
                    # The scheduler is asked to only place a task if the deadline can
                    # be met, and scheduling at this particular start_time leads to a
                    # deadline violation, so the task cannot be placed here.
                    self._space_time_matrix[(worker_id, start_time)] = 0
                elif worker_id not in schedulable_workers:
                    # The Worker cannot accomodate this task, and so the task should
                    # not be placed on this Worker.
                    self._space_time_matrix[(worker_id, start_time)] = 0
                else:
                    # The placement needs to be decided by the optimizer.
                    self._space_time_matrix[
                        (worker_id, start_time)
                    ] = optimizer.binary_var(
                        name=(
                            f"{task.unique_name}_placed_at_Worker"
                            f"_{worker_id}_on_Time_{start_time}"
                        )
                    )

            if task.state == TaskState.SCHEDULED:
                # Maintain a warm-start cache that can be used to pass the starting
                # values to the optimizer.
                placed_key = (
                    self.__get_worker_index_from_previous_placement(task, workers),
                    task.expected_start_time.to(EventTime.Unit.US).time,
                )
                for space_time_index, binary_variable in self._space_time_matrix.keys():
                    if space_time_index == placed_key:
                        self._warm_start_cache[binary_variable] = 1
                    else:
                        self._warm_start_cache[binary_variable] = 0

            if task.state == TaskState.SCHEDULED and not retract_schedules:
                # If the task was previously scheduled, and we do not allow retractions,
                # we allow the start time to be fungible, but the task must be placed.
                optimizer.add_constraint(
                    ct=optimizer.sum(self._space_time_matrix.values()) == 1,
                    ctname=f"{task.unique_name}_previously_scheduled_"
                    f"required_worker_placement",
                )
            else:
                # If either the task was not previously placed, or we are allowing
                # retractions, then the task can be placed or left unplaced.
                optimizer.add_constraint(
                    ct=optimizer.sum(self._space_time_matrix.values()) <= 1,
                    ctname=f"{task.unique_name}_consistent_worker_placement",
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

    def get_placement(
        self,
        solution: SolveSolution,
        worker_index_to_worker: Mapping[int, Worker],
        worker_id_to_worker_pool: Mapping[UUID, UUID],
    ) -> Optional[Placement]:
        """Retrieves the details of the solution from the given `SolveSolution` object,
        and constructs a `Placement` object for the Scheduler to return to the
        Simulator.

        Args:
            solution (`SolveSolution`): The solution computed by the optimizer.
            worker_index_to_worker (`Mapping[int, Worker]`): A mapping from the index
                that the Worker was assigned for this scheduling run to a reference to
                the `Worker` itself.
            worker_id_to_worker_pool (`Mapping[UUID, UUID]`): A mapping from the ID of
                the `Worker` to the ID of the `WorkerPool` which it is a part of.

        Returns:
            A `Placement` object depicting the time when the Task is to be started, and
            the Worker where the Task is to be executed.
        """
        if not solution or self.previously_placed:
            # If there was no solution, or the task was previously placed, then we
            # don't have to return a `Placement` object to the Simulator.
            return None

        for (worker_id, start_time), variable in self._space_time_matrix.items():
            if type(variable) != int and solution.get_value(variable) == 1:
                placement_worker_id = worker_index_to_worker[worker_id].id
                placement_worker_pool_id = worker_id_to_worker_pool[placement_worker_id]
                return Placement(
                    task=self.task,
                    placement_time=EventTime(start_time, unit=EventTime.Unit.US),
                    worker_pool_id=placement_worker_pool_id,
                    worker_id=placement_worker_id,
                )
        return Placement(
            task=self.task, placement_time=None, worker_pool_id=None, worker_id=None
        )

    def get_partition_variable(
        self, time: int, worker_index: int
    ) -> Sequence[Union[cpx_var.Var, int]]:
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
            usage of resources on the given Worker at the given time.
        """
        partition_variables = []
        for (worker_id, start_time), variable in self._space_time_matrix.items():
            if (
                worker_id == worker_index
                and start_time <= time
                and start_time + self.task.remaining_time.to(EventTime.Unit.US).time
                > time
                and (type(variable) == cpx_var.Var or variable == 1)
            ):
                partition_variables.append(variable)
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
    def space_time_matrix(self) -> Mapping[Tuple[int, int], cpx.BinaryVarType]:
        """Returns a mapping from the (Worker Index, Time) to the binary variable
        specifying if the task was placed at that time or not."""
        return self._space_time_matrix


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
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        goal: str = "max_goodput",
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
            lookahead=EventTime.zero(),
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=False,
            _flags=_flags,
        )
        self._goal = goal
        self._gap_time_limit = time_limit
        self._time_discretization = time_discretization
        self._plan_ahead = plan_ahead
        self._log_to_file = log_to_file

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
        if self._log_to_file:
            output_file = open(
                f"./cplex_{current_time.to(EventTime.Unit.US).time}.log", "w"
            )
            optimizer.context.cplex_parameters.mip.display = 5
            optimizer.set_log_output(output_file)

        return (optimizer, output_file)

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: "WorkerPools"
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
            # TODO (Sukrit): Add the Warm Start caches to the problem.
            if self._log_to_file:
                with open(
                    f"./tetrisched_cplex_{sim_time.to(EventTime.Unit.US).time}.lp", "w"
                ) as lp_out:
                    lp_out.write(optimizer.export_as_lp_string())

            # Solve the problem.
            solution: SolveSolution = optimizer.solve()
            if solution:
                # A valid solution was found. Construct the Placement objects.
                self._logger.debug(
                    "[%s] The scheduler returned the objective value %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    solution.objective_value,
                )
                for task_variable in tasks_to_variables.values():
                    # If the task was previously placed,
                    if task_variable.previously_placed:
                        continue
                    placement = task_variable.get_placement(
                        solution, workers, worker_to_worker_pool
                    )
                    if placement.worker_pool_id:
                        self._logger.debug(
                            "[%s] Placed %s (with deadline %s and remaining time "
                            "%s) on WorkerPool(%s) to be started at %s.",
                            sim_time.to(EventTime.Unit.US).time,
                            placement.task.unique_name,
                            placement.task.deadline,
                            placement.task.remaining_time,
                            placement.worker_pool_id,
                            placement.placement_time,
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
                if self._log_to_file:
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
                solution_status: SolveDetails = optimizer.solve_details()
                self._logger.info(
                    "[%s] Failed to place any task because the solver returned %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    solution_status.status,
                )
                for task in tasks_to_be_scheduled:
                    placements.append(
                        Placement(
                            task=task,
                            placement_time=None,
                            worker_pool_id=None,
                            worker_id=None,
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
        tasks_to_variables = {}
        plan_ahead = self._plan_ahead
        if plan_ahead == EventTime(-1, EventTime.Unit.US):
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

    def _add_resource_constraints(
        self,
        sim_time: EventTime,
        optimizer: cpx.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workers: Mapping[int, Worker],
    ):
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
                overlap_variables = {}
                for task_name, task_variable in tasks_to_variables.items():
                    overlap_variables[task_name] = task_variable.get_partition_variable(
                        time=t, worker_index=worker_index
                    )

                # For each resource in the Worker, find the resource request for all
                # the tasks and multiply it by the partition variables to ensure that
                # resources are never oversubscribed.
                for (
                    resource,
                    quantity,
                ) in worker.resources.get_unique_resource_types().items():
                    resource_constraint_terms = []
                    for task_name, task_overlap_variables in overlap_variables.items():
                        task_request_for_resource = tasks_to_variables[
                            task_name
                        ].task.resource_requirements.get_total_quantity(resource)
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
        if self._goal == "max_goodput":
            # Define reward variables for each of the tasks, that is a sum of their
            # space-time matrices. Maximizing the sum of these rewards is the goal of
            # the scheduler.
            task_reward_variables = []
            for task_name, task_variables in tasks_to_variables.items():
                task_reward = optimizer.binary_var(name=f"{task_name}_reward")
                optimizer.add_constraint(
                    ct=(
                        task_reward
                        == optimizer.sum(task_variables.space_time_matrix.values())
                    ),
                    ctname=f"{task_name}_is_placed_reward",
                )
                task_reward_variables.append(task_reward)
            optimizer.maximize(optimizer.sum(task_reward_variables))
        else:
            raise RuntimeError(
                f"The goal {self._goal} is not supported yet by "
                f"TetriSchedCPLEXScheduler."
            )
