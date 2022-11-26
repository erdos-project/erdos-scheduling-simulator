from typing import Mapping, Optional, Sequence

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

        # Placement characteristics.
        self._is_placed = optimizer.addVar(
            vtype=GRB.BINARY, name="f{task.unique_name}_is_placed"
        )
        self._placed_on_worker = None

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

    @property
    def is_placed(self) -> gp.Var:
        return self._is_placed

    @property
    def placed_on_worker(self) -> gp.Var:
        return self._placed_on_worker

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

    def _initialize_resource_constraints(
        self, optimizer: gp.Model, workers: Mapping[int, Worker]
    ) -> None:
        pass

    def initialize_constraints(
        self,
        current_time: EventTime,
        optimizer: gp.Model,
        workers: Mapping[int, Worker],
        enforce_deadlines: bool = True,
    ) -> None:
        self._initialize_timing_constraints(current_time, optimizer, enforce_deadlines)


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

        # Construct the model and the variables for each of the tasks.
        optimizer = gp.Model("ILPScheduler")
        tasks_to_variables = self._add_variables(
            sim_time, optimizer, tasks_to_be_scheduled
        )
        self._add_task_dependency_constraints(optimizer, tasks_to_variables, workload)

        # Add the objectives and return the results.
        self._add_objective(optimizer, tasks_to_variables, workload)
        optimizer.optimize()
        print(f"Found the optimal value: {optimizer.objVal}.")
        print(f"[x] Found values: ")
        for variable in tasks_to_variables.values():
            print(
                f"  [x] {variable.name}: ({variable.is_placed.X}, {variable.start_time.X})"
            )
        print(
            f"Found values: "
            f"{[(variable.is_placed.X, variable.start_time.X) for variable in tasks_to_variables.values()]}."
        )

        raise NotImplementedError(
            "The scheduler has not been finished implementing yet."
        )

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: gp.Model,
        tasks_to_be_scheduled: Sequence[Task],
    ) -> Mapping[str, TaskOptimizerVariables]:
        tasks_to_variables = {}
        for task in tasks_to_be_scheduled:
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                sim_time, task, None, optimizer
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

            # Ensure that the task is started after the last parent has finished.
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

    def _add_objective(
        self,
        optimizer: gp.Model,
        tasks_to_variables: Mapping[str, TaskOptimizerVariables],
        workload: Workload,
    ):
        # TODO (Sukrit): This is wrong. FIX.
        optimizer.setObjective(
            sum(variable.is_placed for variable in tasks_to_variables.values()),
            sense=GRB.MAXIMIZE,
        )
