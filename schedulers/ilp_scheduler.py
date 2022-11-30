from collections import defaultdict
from typing import Mapping, Optional, Sequence, Set

import absl  # noqa: F401
import gurobipy as gp
from gurobipy import GRB

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import BranchPredictionPolicy, Placements, Task, Workload


class TaskOptimizerVariables:
    """TaskOptimizerVariables is used to represent the optimizer variables for
    every particular task to be scheduled by the Scheduler."""

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
        self._start_time = optimizer.addVar(
            vtype=GRB.INTEGER, name=f"{task.unique_name}_start"
        )

        # Placement characteristics
        # Set up individual variables to signify where the task is placed.
        self._placed_on_worker = {}
        for worker_id in workers.keys():
            # TODO (Sukrit): Check if the worker can potentially accomodate the
            # task by itself, if it cannot, we can just set the variable to False.
            placed_on_worker_variable = optimizer.addVar(
                vtype=GRB.BINARY, name=f"{task.unique_name}_placed_on_{worker_id}"
            )
            self._placed_on_worker[worker_id] = placed_on_worker_variable
        self._overlaps_with = {}

        # Initialize the constraints for the variables.
        self.initialize_constraints(current_time, optimizer, enforce_deadlines)

    @property
    def start_time(self) -> gp.Var:
        return self._start_time

    @property
    def task(self) -> Task:
        return self._task

    @property
    def name(self) -> str:
        return self._task.unique_name

    def placed_on_worker(self, worker_id: int) -> Optional[gp.Var]:
        return self._placed_on_worker.get(worker_id)

    @property
    def placed_on_workers(self) -> Sequence[gp.Var]:
        return self._placed_on_worker.values()

    def overlaps_with(self, task_name: str) -> Optional[gp.Var]:
        return self._overlaps_with.get(task_name)

    @property
    def overlaps(self) -> Mapping[str, gp.Var]:
        return self._overlaps_with

    def _initialize_timing_constraints(
        self,
        current_time: EventTime,
        optimizer: gp.Model,
        enforce_deadlines: bool = True,
    ) -> None:
        # Require the task to start atleast at its release time or the current
        # simulator time, whichever one is later.
        optimizer.addConstr(
            self.start_time
            >= max(
                current_time.to(EventTime.Unit.US).time,
                self.task.release_time.to(EventTime.Unit.US).time,
            )
        )

        if enforce_deadlines:
            optimizer.addConstr(
                self.start_time + self.task.remaining_time.to(EventTime.Unit.US).time
                <= self.task.deadline.to(EventTime.Unit.US).time
            )

    def _initialize_placement_constraints(self, optimizer: gp.Model) -> None:
        # Add a constraint to ensure that the task is only placed on a single Worker.
        # We constrain the sum of the individual indicator variables for the placement
        # on a specific Worker to be at most 1.
        # A sum of 0 implies that the task was not placed on any Worker.
        # optimizer.addConstr(gp.quicksum(self._placed_on_worker.values()) <= 1)
        pass

    def initialize_constraints(
        self,
        current_time: EventTime,
        optimizer: gp.Model,
        enforce_deadlines: bool = True,
    ) -> None:
        self._initialize_timing_constraints(current_time, optimizer, enforce_deadlines)
        self._initialize_placement_constraints(optimizer)

    def initialize_overlap_constraints(
        self, optimizer: gp.Model, dependent_tasks: Set[str]
    ) -> None:
        for dependent_task in dependent_tasks:
            overlap_variable = optimizer.addVar(
                vtype=GRB.BINARY, name=f"{self.name}_overlap_{dependent_task}"
            )

            optimizer.addGenConstrIndicator(
                overlap_variable,
                1,
            )


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
        print(
            f"[{sim_time.time}] The scheduler received "
            f"{[task.unique_name for task in tasks_to_be_scheduled]} "
            f"tasks for scheduling."
        )

        # TODO (Sukrit): Reconstruct a worker pool based on the preemptive
        # nature of the scheduler.

        # Construct a mapping of the index of a Worker to the Worker itself.
        worker_index = 1
        workers = {}
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                workers[worker_index] = worker
                worker_index += 1
        print(f"[x] The scheduler constructed {len(workers)} workers for scheduling.")

        # Construct the model and the variables for each of the tasks.
        optimizer = gp.Model("ILPScheduler")
        tasks_to_variables = self._add_variables(
            sim_time, optimizer, tasks_to_be_scheduled, workers
        )
        self._add_task_dependency_constraints(optimizer, tasks_to_variables, workload)
        task_dependency_overlaps = self._add_resource_constraints(
            optimizer, tasks_to_variables, workload, workers
        )

        # Add the objectives and return the results.
        self._add_objective(optimizer, tasks_to_variables, workload)
        optimizer.optimize()
        print(f"Found the optimal value: {optimizer.objVal}.")
        print(f"[x] Found values: ")
        for variable in tasks_to_variables.values():
            print(
                f"  [x] {variable.name}: Start Time = {variable.start_time.X}, "
                f"Deadline = {variable.task.deadline}"
            )
            for _, dependency_variable in task_dependency_overlaps[variable.name]:
                print(
                    f"      [x] {dependency_variable.VarName}: {dependency_variable.X}"
                )
            for worker_id in range(1, worker_index):
                print(
                    f"      [x] Placed on Worker {worker_id}: {variable.placed_on_worker(worker_id).X}."
                )

        raise NotImplementedError(
            "The scheduler has not been finished implementing yet."
        )

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        tasks_to_be_scheduled: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
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
        for variable in tasks_to_variables.values():
            task = variable.task
            print(f"[x] Working for {task.unique_name}.")
            task_graph = workload.get_task_graph(variable.task.task_graph)
            parent_variables = [
                tasks_to_variables[parent.unique_name]
                for parent in task_graph.get_parents(task)
                if parent.unique_name in tasks_to_variables
            ]

            # Ensure that the task is only placed if all of its parents are placed,
            # and that it is started after the last parent has finished.
            all_parents_placed = optimizer.addVar(
                vtype=GRB.BINARY, name=f"{variable.name}_all_parents_placed"
            )
            parent_placements = []
            for parent_variable in parent_variables:
                print(
                    f"  [x] Remaining Time for parent "
                    f"({parent_variable.name}): "
                    f"{parent_variable.task.remaining_time.time}."
                )
                optimizer.addConstr(
                    variable.start_time
                    >= parent_variable.start_time
                    + parent_variable.task.remaining_time.to(EventTime.Unit.US).time
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
            )
            optimizer.addGenConstrIndicator(
                all_parents_placed,
                1,
                gp.quicksum(parent_placements),
                GRB.EQUAL,
                len(parent_variables),
            )

            # If all of the parents were not placed, then we cannot place this task.
            optimizer.addGenConstrIndicator(
                all_parents_placed,
                0,
                gp.quicksum(variable.placed_on_workers),
                GRB.EQUAL,
                0,
            )

    def _add_resource_constraints(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
        workers: Mapping[int, Worker],
    ) -> None:
        # Find all tasks that might run in parallel when a particular task is placed.
        task_resource_dependencies: Mapping[str, str] = defaultdict(set)
        for task_name_1, variable_1 in tasks_to_variables.items():
            for task_name_2, variable_2 in tasks_to_variables.items():
                if (
                    task_name_1 == task_name_2
                    or len(variable_1.placed_on_workers) == 0
                    or len(variable_2.placed_on_workers) == 0
                ):
                    # Either same task or one of the tasks cannot be placed.
                    continue

                if variable_1.task.task_graph == variable_2.task.task_graph:
                    # If the tasks are in the same graph, and they are not dependent
                    # on each other, they can run in parallel, and hence, must be
                    # checked for overlaps.
                    task_graph = workload.get_task_graph(variable_1.task.task_graph)
                    if not task_graph.are_dependent(variable_1.task, variable_2.task):
                        task_resource_dependencies[task_name_1].add(task_name_2)
                else:
                    # If the tasks belong to different graphs, they can always run in
                    # parallel, and hence, must be checked for overlaps.
                    task_resource_dependencies[task_name_1].add(task_name_2)

        # For all of the tasks, specify variables to check if the task is running
        # when the particular task is started.
        task_dependency_overlaps = defaultdict(list)
        for task, task_dependencies in task_resource_dependencies.items():
            print(f"[x] Dependencies of {task_name_1}: {task_dependencies}.")
            for task_dependency in task_dependencies:
                task_dependency_overlaps[task].append(
                    (
                        task_dependency,
                        self._overlaps(
                            optimizer,
                            tasks_to_variables[task],
                            tasks_to_variables[task_dependency],
                        ),
                    )
                )

        # We now ensure that for all the Tasks along with their potential overlap
        # dependencies, whatever subset of the Tasks are placed on the same Worker
        # do not end up oversubscribing the Worker's resources.
        for task, task_dependencies in task_dependency_overlaps.items():
            for worker_index, worker in workers.items():
                for (
                    resource,
                    quantity,
                ) in worker.resources.get_unique_resource_types().items():
                    task_request_for_resource = tasks_to_variables[
                        task
                    ].task.resource_requirements.get_total_quantity(resource)
                    if quantity == 0 or task_request_for_resource == 0:
                        # We ensure earlier that the Worker has enough space to accomodate
                        # each task. If the quantity of the resource is 0 or the task does
                        # not need this resource, then we can skip the addition of a
                        # constraint since the task would not have been placed on this
                        # worker or taken up this resource.
                        continue

                    # If the dependency overlaps and the dependency is placed on this
                    # worker, then ensure that the resource type is not oversubscribed.
                    resource_constraint_expression = gp.QuadExpr(
                        tasks_to_variables[task].placed_on_worker(worker_index)
                        * task_request_for_resource
                    )
                    for dependent_task, overlap_variable in task_dependencies:
                        dependent_task_request_for_resource = tasks_to_variables[
                            dependent_task
                        ].task.resource_requirements.get_total_quantity(resource)
                        if dependent_task_request_for_resource == 0:
                            continue
                        resource_constraint_expression.add(
                            tasks_to_variables[dependent_task].placed_on_worker(
                                worker_index
                            )
                            * overlap_variable
                            * dependent_task_request_for_resource
                        )
                    optimizer.addConstr(resource_constraint_expression <= quantity)
        return task_dependency_overlaps

    def _overlaps(
        self,
        optimizer: gp.Model,
        task_1: TaskOptimizerVariables,
        task_2: TaskOptimizerVariables,
    ) -> gp.Var:
        task_1_ends_before_task_2_starts = optimizer.addVar(
            vtype=GRB.BINARY,
            name=f"{task_1.name}_ends_before_{task_2.name}_starts",
        )
        task_2_ends_before_task_1_starts = optimizer.addVar(
            vtype=GRB.BINARY,
            name=f"{task_2.name}_ends_before_task_2_starts",
        )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            0,
            task_1.start_time
            + task_1.task.remaining_time.to(EventTime.Unit.US).time
            - task_2.start_time,
            GRB.GREATER_EQUAL,
            0,
        )
        optimizer.addGenConstrIndicator(
            task_1_ends_before_task_2_starts,
            1,
            task_1.start_time
            + task_1.task.remaining_time.to(EventTime.Unit.US).time
            - task_2.start_time,
            GRB.LESS_EQUAL,
            -1,
        )
        optimizer.addGenConstrIndicator(
            task_2_ends_before_task_1_starts,
            0,
            task_1.start_time
            - task_2.start_time
            - task_2.task.remaining_time.to(EventTime.Unit.US).time,
            GRB.LESS_EQUAL,
            0,
        )
        optimizer.addGenConstrIndicator(
            task_2_ends_before_task_1_starts,
            1,
            task_1.start_time
            - task_2.start_time
            - task_2.task.remaining_time.to(EventTime.Unit.US).time,
            GRB.GREATER_EQUAL,
            1,
        )
        task_1_task_2_overlap = optimizer.addVar(
            vtype=GRB.BINARY, name=f"{task_1.name}_overlaps_{task_2.name}"
        )
        # TODO (Sukrit): This constraint below should work.
        optimizer.addConstr(
            task_1_ends_before_task_2_starts
            + task_2_ends_before_task_1_starts
            + task_1_task_2_overlap
            == 1
        )
        return task_1_task_2_overlap

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        # TODO (Sukrit): This is wrong. FIX.
        placed_on_workers = []
        for task_variable in tasks_to_variables.values():
            placed_on_workers.extend(task_variable.placed_on_workers)

        optimizer.setObjective(gp.quicksum(placed_on_workers), sense=GRB.MAXIMIZE)
