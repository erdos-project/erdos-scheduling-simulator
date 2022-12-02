import time
from collections import defaultdict
from copy import copy, deepcopy
from itertools import combinations
from typing import Mapping, Optional, Sequence, Union

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import BranchPredictionPolicy, Placement, Placements, Task, Workload


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
    ):
        self._task = task

        # Timing characteristics.
        if enforce_deadlines:
            self._start_time = optimizer.addVar(
                lb=max(
                    current_time.to(EventTime.Unit.US).time,
                    task.release_time.to(EventTime.Unit.US).time,
                ),
                ub=(
                    task.deadline.to(EventTime.Unit.US).time
                    - task.remaining_time.to(EventTime.Unit.US).time
                ),
                vtype=GRB.INTEGER,
                name=f"{task.unique_name}_start",
            )
        else:
            self._start_time = optimizer.addVar(
                lb=max(
                    current_time.to(EventTime.Unit.US).time,
                    task.release_time.to(EventTime.Unit.US).time,
                ),
                vtype=GRB.INTEGER,
                name=f"{task.unique_name}_start",
            )

        # Placement characteristics
        # Set up individual variables to signify where the task is placed.
        self._placed_on_worker = {}
        for worker_id, worker in workers.items():
            # If the worker cannot accomodate the task, we set the placement variable
            # to 0 to reduce the number of constraints in the system.
            if worker.can_accomodate_task(task):
                self._placed_on_worker[worker_id] = optimizer.addVar(
                    vtype=GRB.BINARY, name=f"{task.unique_name}_placed_on_{worker.name}"
                )
            else:
                self._placed_on_worker[worker_id] = 0

        # Initialize the constraints for the variables.
        self.initialize_constraints(optimizer)

    @property
    def start_time(self) -> gp.Var:
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

    def placed_on_worker(self, worker_id: int) -> Optional[Union[int, gp.Var]]:
        """Check if the Task was placed on a particular Worker.

        Args:
            worker_id (`int`): The ID of the Worker to check.

        Returns:
            The method has the following return values:
                - `None`: If the `Worker` ID was not registered with the instance.
                - `int`: If the `Worker` was forced to not be placed on this Worker.
                - `gp.Var`: A Gurobi variable representing the placement solution.
        """
        return self._placed_on_worker.get(worker_id)

    @property
    def placed_on_workers(self) -> Sequence[gp.Var]:
        """Retrieves the binary Gurobi variables representing the placement of the
        task on the collection of `Worker`s registered with the instance."""
        return self._placed_on_worker.values()

    def __str__(self) -> str:
        return f"TaskOptimizerVariables(name={self.name})"

    def __repr__(self) -> str:
        return str(self)

    def _initialize_placement_constraints(self, optimizer: gp.Model) -> None:
        # Add a constraint to ensure that the task is only placed on a single Worker.
        # We constrain the sum of the individual indicator variables for the placement
        # on a specific Worker to be at most 1.
        # A sum of 0 implies that the task was not placed on any Worker.
        optimizer.addConstr(
            gp.quicksum(self._placed_on_worker.values()) <= 1,
            name=f"{self.name}_consistent_worker_placement",
        )

    def initialize_constraints(
        self,
        optimizer: gp.Model,
    ) -> None:
        """Initializes the constraints for the particular `Task`.

        Args:
            optimizer (`gp.Model`): The Gurobi model to which the constraints must
                be added.
        """
        self._initialize_placement_constraints(optimizer)


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
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
        goal (`str`): The goal to use as the optimization objective.
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
        retract_schedules: bool = False,
        goal: str = "max_goodput",
        _flags: Optional["absl.flags"] = None,
    ):
        super(ILPScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            policy=policy,
            retract_schedules=retract_schedules,
            _flags=_flags,
        )
        self._goal = goal

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
            release_taskgraphs=self.release_taskgraphs,
        )

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
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)}"
            f"tasks for scheduling across {len(workers)} workers."
        )

        # Construct the model and the variables for each of the tasks.
        scheduler_start_time = time.time()
        optimizer = gp.Model("ILPScheduler")
        tasks_to_variables = self._add_variables(
            sim_time, optimizer, tasks_to_be_scheduled, workers
        )
        self._add_task_dependency_constraints(optimizer, tasks_to_variables, workload)
        self._add_resource_constraints(
            sim_time, optimizer, tasks_to_variables, workload, workers
        )

        # Add the objectives and optimize the model.
        self._add_objective(optimizer, tasks_to_variables, workload)
        optimizer.optimize()
        optimizer.write("test_file.lp")
        self._logger.info(
            f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned the "
            f"status {optimizer.status}."
        )

        # Collect the placement results.
        placements = []
        if optimizer.Status == GRB.OPTIMAL:
            self._logger.info(
                f"[{sim_time.to(EventTime.Unit.US).time}] The scheduler returned the "
                f"objective value {optimizer.objVal}."
            )
            for task_name, variables in tasks_to_variables.items():
                # Find the Worker where the Task was placed.
                worker_pool_id = None
                for worker_id, worker in workers.items():
                    if isinstance(variables.placed_on_worker(worker_id), gp.Var):
                        placement = variables.placed_on_worker(worker_id).X
                        if placement == 1:
                            worker_pool_id = worker_to_worker_pool[worker.id]

                # If the task was placed, find the start time.
                if worker_pool_id is not None:
                    start_time = int(variables.start_time.X)
                    placements.append(
                        Placement(
                            variables.task,
                            worker_pool_id,
                            EventTime(start_time, EventTime.Unit.US),
                        )
                    )
                    self._logger.debug(
                        "[%s] Placed %s on WorkerPool(%s) to be started at %d.",
                        sim_time.to(EventTime.Unit.US).time,
                        task_name,
                        worker_pool_id,
                        start_time,
                    )
                else:
                    placements.append(Placement(variables.task))
                    self._logger.debug(
                        "[%s] Failed to place %s because no WorkerPool "
                        "could accomodate the resource requirements.",
                        sim_time.to(EventTime.Unit.US).time,
                        task_name,
                    )

        else:
            for task_name, variables in tasks_to_variables.items():
                placements.append(Placement(variables.task))
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
        return Placements(runtime, placements)

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
        tasks_to_variables = {}
        for task in tasks_to_be_scheduled:
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                sim_time, task, workers, optimizer, self.enforce_deadlines
            )
        return tasks_to_variables

    def _add_task_dependency_constraints(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
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
        """
        for task_name, variable in tasks_to_variables.items():
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
                    optimizer.addConstr(
                        variable.start_time
                        >= parent_variable.start_time
                        + parent_variable.task.remaining_time.to(EventTime.Unit.US).time
                        + 1,
                        name=f"{task_name}_start_after_{parent_variable.name}",
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
        current_time: EventTime,
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
        # Construct variables for all the possible task pairs.
        task_pairs = list(combinations(tasks_to_variables.keys(), r=2))
        task_pair_overlap_variables = optimizer.addVars(
            task_pairs, vtype=GRB.BINARY, name="Overlap"
        )
        optimizer.update()

        # For each pair, set up constraints for the overlap variables.
        for task_1_name, task_2_name in task_pairs:
            task_1_variable = tasks_to_variables[task_1_name]
            task_2_variable = tasks_to_variables[task_2_name]
            overlap_variables = task_pair_overlap_variables.select(
                task_1_name, task_2_name
            )
            if len(overlap_variables) != 1:
                raise ValueError(
                    f"Expected only one Overlap variable for ({task_1_name}, "
                    "{task_2_name}). Found {len(overlap_variables)}."
                )
            task_pair_overlap_variable = overlap_variables[0]

            # If one of the tasks cannot be placed, force the overlap
            # variable to be set to 0 during the preprocessing step.
            if (
                len(task_1_variable.placed_on_workers) == 0
                or len(task_2_variable.placed_on_workers) == 0
            ):
                self._logger.debug(
                    "[%s] Forcing overlap of %s and %s to be 0 since the length "
                    "of their available workers was %d and %d respectively.",
                    current_time.to(EventTime.Unit.US).time,
                    task_1_name,
                    task_2_name,
                    len(task_1_variable.placed_on_workers),
                    len(task_2_variable.placed_on_workers),
                )
                optimizer.addConstr(
                    task_pair_overlap_variable == 0,
                    name=f"{task_1_name}_no_overlap_{task_2_name}_cannot_place",
                )
                continue

            # If the tasks belong to the same graph, then they need to be checked
            # for overlap, otherwise, the two tasks can always overlap.
            if task_1_variable.task.task_graph == task_2_variable.task.task_graph:
                task_graph = workload.get_task_graph(task_1_variable.task.task_graph)
                if task_graph.are_dependent(task_1_variable.task, task_2_variable.task):
                    # If the tasks are dependent on each other, they can never overlap.
                    self._logger.debug(
                        "[%s] Forcing overlap of %s and %s to be 0 since they "
                        "are dependent tasks of the TaskGraph %s.",
                        current_time.to(EventTime.Unit.US).time,
                        task_1_name,
                        task_2_name,
                        task_1_variable.task.task_graph,
                    )
                    optimizer.addConstr(
                        task_pair_overlap_variable == 0,
                        name=f"{task_1_name}_no_overlap_{task_2_name}_dependent",
                    )
                else:
                    # If the tasks are not dependent on each other, they may overlap.
                    self._overlaps(
                        optimizer,
                        task_1_variable,
                        task_2_variable,
                        task_pair_overlap_variable,
                    )
            else:
                self._overlaps(
                    optimizer,
                    task_1_variable,
                    task_2_variable,
                    task_pair_overlap_variable,
                )

        # We now ensure that for all the tasks, their overlap dependencies don't
        # end up oversubscribing any Worker's resources.
        for task_name in tasks_to_variables:
            # Find all the dependencies of this task that can overlap with it.
            task_dependencies = []
            for (
                task_1_name,
                task_2_name,
            ), variable in task_pair_overlap_variables.items():
                if task_1_name == task_name:
                    task_dependencies.append((task_2_name, variable))
                elif task_2_name == task_name:
                    task_dependencies.append((task_1_name, variable))

            # For each Worker and each of its resources, ensure no oversubscription.
            for worker_index, worker in workers.items():
                for (
                    resource,
                    quantity,
                ) in worker.resources.get_unique_resource_types().items():
                    task_request_for_resource = tasks_to_variables[
                        task_name
                    ].task.resource_requirements.get_total_quantity(resource)
                    if quantity == 0 or task_request_for_resource == 0:
                        # We ensure earlier that the Worker has enough space to
                        # accomodate # each task. If the quantity of the resource is
                        # 0 or the task does not need this resource, then we can
                        # skip the addition of a constraint since the task would not
                        # have been placed on this worker or taken up this resource.
                        continue

                    # If the dependency overlaps and the dependency is placed on this
                    # worker, then ensure that the resource type is not oversubscribed.
                    resource_constraint_expression = gp.QuadExpr(
                        tasks_to_variables[task_name].placed_on_worker(worker_index)
                        * task_request_for_resource
                    )
                    for dependent_task_name, overlap_variable in task_dependencies:
                        dependent_task_request_for_resource = tasks_to_variables[
                            dependent_task_name
                        ].task.resource_requirements.get_total_quantity(resource)
                        if dependent_task_request_for_resource == 0:
                            # The dependent task's request for this resource was empty.
                            # It cannot contend with the current task on this resource.
                            continue
                        resource_constraint_expression.add(
                            tasks_to_variables[dependent_task_name].placed_on_worker(
                                worker_index
                            )
                            * overlap_variable
                            * dependent_task_request_for_resource
                        )

                    # Add the constraint to the optimizer.
                    optimizer.addConstr(
                        resource_constraint_expression <= quantity,
                        name=f"{task_name}_{worker.name}_{resource.name}_constraint",
                    )

    def _overlaps(
        self,
        optimizer: gp.Model,
        task_1: TaskOptimizerVariables,
        task_2: TaskOptimizerVariables,
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
        # Add an indicator variable that checks if the first task ends before the
        # second task starts.
        task_1_ends_before_task_2_starts = optimizer.addVar(
            vtype=GRB.BINARY,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts",
        )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            0,
            task_1.start_time
            + task_1.task.remaining_time.to(EventTime.Unit.US).time
            - task_2.start_time,
            GRB.GREATER_EQUAL,
            0,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts_False",
        )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            1,
            task_1.start_time
            + task_1.task.remaining_time.to(EventTime.Unit.US).time
            - task_2.start_time,
            GRB.LESS_EQUAL,
            -1,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts_True",
        )

        # Add an indicator variable that checks if the second task ends before the
        # first task starts.
        task_2_ends_before_task_1_starts = optimizer.addVar(
            vtype=GRB.BINARY,
            name=f"{task_2.name}_ends_before_{task_1.name}_starts",
        )
        optimizer.addGenConstrIndicator(
            task_2_ends_before_task_1_starts,
            0,
            task_1.start_time
            - task_2.start_time
            - task_2.task.remaining_time.to(EventTime.Unit.US).time,
            GRB.LESS_EQUAL,
            0,
            name=f"{task_2.name}_ends_before_{task_1.name}_starts_False",
        )
        optimizer.addGenConstrIndicator(
            task_2_ends_before_task_1_starts,
            1,
            task_1.start_time
            - task_2.start_time
            - task_2.task.remaining_time.to(EventTime.Unit.US).time,
            GRB.GREATER_EQUAL,
            1,
            name=f"{task_2.name}_ends_before_{task_1.name}_starts_True",
        )

        # Add an indicator variable that is set to 1 if neither of the above
        # options can be evaluted to 1.
        if overlap_variable is None:
            overlap_variable = optimizer.addVar(
                vtype=GRB.BINARY, name=f"Overlap({task_1.name}, {task_2.name})"
            )
        optimizer.addConstr(
            task_1_ends_before_task_2_starts
            + task_2_ends_before_task_1_starts
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
    ):
        # TODO (Sukrit): This is wrong. FIX.
        placed_on_workers = []
        for task_variable in tasks_to_variables.values():
            if workload.get_task_graph(task_variable.task.task_graph).is_sink_task(
                task_variable.task
            ):
                placed_on_workers.extend(task_variable.placed_on_workers)

        optimizer.setObjective(gp.quicksum(placed_on_workers), sense=GRB.MAXIMIZE)
