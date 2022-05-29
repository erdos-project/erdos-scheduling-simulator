import functools
import pickle
import sys
import time
from collections import defaultdict
from operator import attrgetter
from typing import Optional

import absl  # noqa: F401
import gurobipy as gp

import utils
from schedulers import BaseScheduler
from workers import WorkerPools
from workload import Resource, Resources, TaskGraph, TaskState

# Minimum task gain constant, which is used to set the benefit of leaving a task
# unscheduled. It is equivalent to missing a task deadline by 3 seconds.
MIN_TASK_GAIN = -3 * 10**6


class GurobiBaseScheduler(BaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        goal: str = "max_slack",
        enforce_deadlines: bool = True,
        lookahead: int = 0,
        _flags: Optional["absl.flags"] = None,
    ):
        """Constructs a Gurobi scheduler.

        Args:
            preemptive (`bool`): If `True`, the Gurobi scheduler can preempt
                the tasks that are currently running.
            runtime (`int`): The runtime to return to the simulator (in us).
                If -1, the scheduler returns the actual runtime.
            goal (`str`): Goal of the scheduler run.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                schedule will return None.
            lookahead (`int`): The scheduler will try to place tasks that are within
                the scheduling lookahead (in us) using estimated task release times.
        """
        super(GurobiBaseScheduler, self).__init__(
            preemptive, runtime, lookahead, enforce_deadlines, _flags
        )
        self._goal = goal
        self._time = None
        self._task_ids_to_task = {}
        # Mapping from task id to the var storing the task start time.
        self._task_ids_to_start_time = {}
        self._task_graph = None
        self._worker_pools = None
        self._placements = []

    def _add_task_timing_constraints(self):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                self._model.addConstr(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time. No need to add this constraint because
            # we set a lower bound on the start time variables.
            # self._model.addConstr(task.release_time <= start_time)

    def _add_task_dependency_constraints(self):
        for task_id, task in self._task_ids_to_task.items():
            for child_task in self._task_graph.get_children(task):
                # Dependent tasks need to finish before the next one.
                # If task does not have a start_time variable, then it is estimated
                # that the task will not be ready to executing within the lookahead.
                # Similarly, if the parent task does not complete within lookahead,
                # then the child task does not have a start_time variable.
                if child_task.id in self._task_ids_to_start_time:
                    self._model.addConstr(
                        self._task_ids_to_start_time[task_id] + task.remaining_time
                        <= self._task_ids_to_start_time[child_task.id]
                    )

    def _in_interval_slow(self, event_time, start_time, end_time, var_name=None):
        if var_name is None:
            var_name = f"time_{event_time}_in_{start_time},{end_time}"
        in_var = self._model.addVar(vtype=gp.GRB.BINARY, name=var_name)
        or_var = self._model.addVar(vtype=gp.GRB.BINARY, name="or_" + var_name)
        # If in_var == 1 then start_time <= event_time <= end_time.
        self._model.addConstr((in_var == 1) >> (event_time >= start_time))
        self._model.addConstr((in_var == 1) >> (event_time <= end_time))
        M = 2 * 1000 * 1000
        # If in_var == 0 & or_var == 0 then event_time <= start_time - 1
        self._model.addConstr(
            (in_var == 0) >> (event_time <= start_time - 1 + M * or_var)
        )
        # If in_var == 0 & or_var == 1 then event_time >= end_time + 1
        self._model.addConstr(
            (in_var == 0) >> (event_time >= end_time + 1 - M * (1 - or_var))
        )
        return in_var

    def _in_interval(self, event_time, start_time, end_time, var_name=None):
        """Adds constraints to check if event time is in [start, end] time interval.

        Args:
            event_time: The time of the event to check if is in the interval.
            start_time: The start of the time interval.
            end_time: The end of the time interval.
            var_name: Name to give to the Gurobi binary variable.

        Returns:
            A Gurobi variable that is 1 if the time is in the interval, 0 otherwise.
        """
        if var_name is None:
            var_name = f"time_{event_time}_in_{start_time},{end_time}"
        in_vars = self._model.addVars(3, vtype=gp.GRB.BINARY, name=var_name)
        self._model.addConstr((in_vars[0] == 1) >> (event_time >= start_time))
        self._model.addConstr((in_vars[0] == 1) >> (event_time <= end_time))
        self._model.addConstr((in_vars[1] == 1) >> (event_time <= start_time - 1))
        self._model.addConstr((in_vars[2] == 1) >> (event_time >= end_time + 1))
        self._model.addConstr(gp.quicksum(in_vars) == 1)
        return in_vars[0]

    def _in_interval_approximate(self, event_time, start_time, end_time):
        if start_time <= event_time and event_time <= end_time:
            return 1
        return 0

    def schedule(self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools):
        self._initialize_state(sim_time, task_graph, worker_pools)
        scheduler_start_time = time.time()

        self._add_variables()
        self._add_task_timing_constraints()
        self._add_task_dependency_constraints()
        self._add_task_resource_constraints()
        self._add_objective()

        self._model.optimize()
        scheduler_end_time = time.time()
        runtime = int((scheduler_end_time - scheduler_start_time) * 10**6)
        self._logger.info(f"Scheduler wall-clock runtime: {runtime}")
        if self.runtime != -1:
            runtime = self.runtime

        if self._model.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {self._model.objVal}")
            self._placements = []
            self._gain = round(self._model.objVal)
            for task_id, task in self._task_ids_to_task.items():
                start_time = round(self._task_ids_to_start_time[task.id].X)
                placement = self._get_task_placement(task)
                if start_time <= sim_time + runtime * 2:
                    # We only place the tasks with a start time earlier than
                    # the estimated end time of the next scheduler run.
                    # Therefore, a task can progress before the next scheduler
                    # finishes. However, the next scheduler will assume that
                    # the task is not running while considering for placement.
                    self._placements.append((task, placement, start_time))
                else:
                    self._placements.append((task, None, None))
            self.log()
            self._verify_schedule(worker_pools, self._task_graph, self._placements)
        else:
            self._placements = [
                (task, None, None) for task in self._task_ids_to_task.values()
            ]
            self._gain = -sys.maxsize
            if self._model.status == gp.GRB.INFEASIBLE:
                self._logger.debug("Solver couldn't find a solution.")
                self.log_solver_internal()
            else:
                self._logger.debug(f"Solver failed with status: {self._model.status}")
                self.log_solver_internal()
            self.log()
            raise ValueError("Model should never be infeasible")
        return runtime, self._placements

    def log_solver_internal(self):
        self._logger.debug(f"Solver stats: {self._model.printStats()}")
        self._model.computeIIS()
        self._logger.debug("The following constraint(s) cannot be satisfied:")
        for c in self._model.getConstrs():
            if c.IISConstr:
                self._logger.debug(f"Constraint: {c}")

    def log(self):
        if self._flags is not None and self._flags.scheduler_log_base_name is not None:
            with open(self._flags.scheduler_log_base_name + ".pkl", "wb") as log_file:
                logged_data = {
                    "time": self._time,
                    "tasks": self._task_ids_to_task,
                    "task_graph": self._task_graph,
                    # "worker_pools": self._worker_pools,
                    "lookahead": self._lookahead,
                    "runtime": self.runtime,
                    "placements": self._placements,
                    "gain": self._gain,
                }
                pickle.dump(logged_data, log_file)
            self._model.write(self._flags.scheduler_log_base_name + ".sol")
            self._model.write(self._flags.scheduler_log_base_name + ".lp")


class GurobiScheduler(GurobiBaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        goal: str = "max_slack",
        enforce_deadlines: bool = True,
        lookahead: int = 0,
        _flags: Optional["absl.flags"] = None,
    ):
        super(GurobiScheduler, self).__init__(
            preemptive, runtime, goal, enforce_deadlines, lookahead, _flags
        )
        # Mapping from task id to the var storing the task resource placement.
        self._task_ids_to_resources = defaultdict(list)
        # Mapping from task id to the variable storing the gain of the placement.
        self._task_ids_to_gain = {}
        # Mapping from task id to the variable controlling if the task is placed
        # or not.
        self._task_ids_to_scheduled_flag = {}

    def _initialize_state(
        self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools
    ):
        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.lookahead, self.preemptive, worker_pools
        )
        self._time = sim_time
        # Rest the state.
        self._task_ids_to_task = {}
        for task in tasks:
            self._task_ids_to_task[task.id] = task
        self._task_ids_to_start_time = {}
        self._task_ids_to_resources = defaultdict(list)
        self._task_ids_to_gain = {}
        self._task_graph = task_graph
        self._worker_pools = worker_pools
        self._model = gp.Model("RAP")
        self._model.Params.LogToConsole = 0
        self._model.Params.OptimalityTol = 0.0005
        self._model.Params.IntFeasTol = 0.00001
        # self._model.Params.TimeLimit = 1  # In seconds.
        # Sets the solver method to concurrent and deterministic.
        # self._model.Params.Method = 4
        (
            self._res_type_index_range,
            self._res_index_to_wp_id,
            self._res_index_to_wp_index,
            self._num_workers,
        ) = worker_pools.get_resource_ilp_encoding()

    def _add_variables(self):
        num_resources = len(self._res_index_to_wp_index.keys())
        # We are solving for start_times and placements while maximizin gains.
        for task in self._task_ids_to_task.values():
            # Add a variable to store the gain of the task assignment.
            self._task_ids_to_gain[task.id] = self._model.addVar(
                vtype=gp.GRB.INTEGER,
                lb=MIN_TASK_GAIN,
                ub=task.deadline
                - self._time
                - task.remaining_time,  # Task cannot start before sim time.
                name=f"gain_task_{task.unique_name}",
            )
            # Add a variable to store the start time of the task.
            self._task_ids_to_start_time[task.id] = self._model.addVar(
                vtype=gp.GRB.INTEGER,
                lb=max(
                    self._time, task.release_time
                ),  # Start times cannot be less than sim time.
                name=f"start_time_task_{task.unique_name}",
            )
            # Add a variable to control if the task is placed or left unplaced.
            # A value of 0 means that the task is be placed, and a value of 1
            # means that the task is not placed.
            # Note: Do not change the encoding as the variable is also used to
            # ensure that two tasks do not overlap on the same resource.
            scheduled_flag_var = self._model.addVar(
                vtype=gp.GRB.BINARY, name=f"scheduled_task_{task.unique_name}"
            )
            self._task_ids_to_scheduled_flag[task.id] = scheduled_flag_var
            prev_worker_var = None
            for (
                resource,
                quantity,
            ) in task.resource_requirements._resource_vector.items():
                # For each unit required we create resource and worker variables.
                for index in range(quantity):
                    # Stores the id of the resource that got allocated to the task.
                    res_var = self._model.addVar(
                        vtype=gp.GRB.INTEGER,
                        lb=-1,
                        ub=num_resources - 1,
                        name=f"resource_task_{task.unique_name}_"
                        f"{resource.name}_{index}",
                    )
                    # Stores the id of worker containing the resource that got
                    # allocated to the task.
                    worker_var = self._model.addVar(
                        vtype=gp.GRB.INTEGER,
                        lb=-1,
                        ub=self._num_workers - 1,
                        name=f"worker_task_{task.unique_name}_{resource.name}_{index}",
                    )
                    self._task_ids_to_resources[task.id].append(res_var)
                    # Add constraints whether a task is placed on GPU or CPU.
                    (start_index, end_index) = self._res_type_index_range[resource.name]
                    # If the task is not scheduled, then its resource and worker
                    # variables must be -1.
                    self._model.addConstr((scheduled_flag_var == 1) >> (res_var == -1))
                    self._model.addConstr(
                        (scheduled_flag_var == 1) >> (worker_var == -1)
                    )
                    self._model.addConstr(
                        (scheduled_flag_var == 0) >> (res_var >= start_index)
                    )
                    self._model.addConstr(
                        (scheduled_flag_var == 0) >> (res_var <= end_index - 1)
                    )
                    # if ri == resource_index => worker_var == worker_index, where
                    # worker_index is the index of the worker the resource in part of.
                    for res_index in range(start_index, end_index):
                        res_diff_var = self._model.addVar(
                            vtype=gp.GRB.INTEGER,
                            lb=-num_resources,
                            ub=num_resources,
                            name=f"resource_diff_task_{task.unique_name}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        abs_diff_var = self._model.addVar(
                            lb=0,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"abs_resource_diff_task_{task.unique_name}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        flag_var = self._model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"flag_resource_diff_task_{task.unique_name}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        self._model.addConstr(res_diff_var == res_var - res_index)
                        self._model.addConstr(abs_diff_var == gp.abs_(res_diff_var))
                        self._model.addConstr((flag_var == 1) >> (abs_diff_var >= 1))
                        # If the resource var if equal to the index then its
                        # corresponding worker variable must equal to the worker index
                        # of the resource.
                        self._model.addConstr(
                            (flag_var == 0)
                            >> (worker_var == self._res_index_to_wp_index[res_index])
                        )
                        self._model.addConstr((flag_var == 0) >> (abs_diff_var == 0))

                    # A task must receive all its resources on a single worker.
                    if prev_worker_var:
                        self._model.addConstr(prev_worker_var == worker_var)
                    prev_worker_var = worker_var

    def _add_task_resource_constraints(self):
        num_resources = len(self._res_index_to_wp_index.keys())
        # Tasks using the same resources must not overlap.
        if len(self._task_ids_to_task) == 0:
            return
        max_deadline = max([task.deadline for task in self._task_ids_to_task.values()])
        tasks = self._task_ids_to_task.items()
        # For every task pair we need to ensure that if the tasks utilize the same
        # resource then their execution does not overlap.
        for index1, (t1_id, task1) in enumerate(tasks):
            for index2, (t2_id, task2) in enumerate(tasks):
                if index2 > index1:
                    # We only need to place constraints for each pair of tasks once.
                    # Note: We also place constraints among resources used by a task.
                    break
                resources_1 = self._task_ids_to_resources[t1_id]
                resources_2 = self._task_ids_to_resources[t2_id]
                for r1_index, t1_res in enumerate(resources_1):
                    for r2_index, t2_res in enumerate(resources_2):
                        if index1 == index2 and r2_index >= r1_index:
                            break
                        diff_var = self._model.addVar(
                            lb=-num_resources,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"res_diff_task_{task1.unique_name}_"
                            f"res_index_{r1_index}_task_{task2.unique_name}_"
                            f"res_index_{r2_index}",
                        )
                        abs_diff_var = self._model.addVar(
                            lb=0,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"abs_res_diff_task_{task1.unique_name}_"
                            f"res_index_{r1_index}_task_{task2.unique_name}_"
                            f"res_index_{r2_index}",
                        )
                        flag_var = self._model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"flag_task_{task1.unique_name}_res_index_{r1_index}_"
                            f"task_{task2.unique_name}_res_index_{r2_index}",
                        )
                        or_var = self._model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"or_task_{task1.unique_name}_res_index_{r1_index}_"
                            f"task_{task2.unique_name}_res_index_{r2_index}",
                        )
                        self._model.addConstr(diff_var == t1_res - t2_res)
                        self._model.addConstr(abs_diff_var == gp.abs_(diff_var))
                        # Ensure that either the tasks do not use the same resource,
                        # or at least one of them was left unscheduled.
                        self._model.addConstr(
                            (flag_var == 1)
                            >> (
                                abs_diff_var
                                + self._task_ids_to_scheduled_flag[t1_id]
                                + self._task_ids_to_scheduled_flag[t2_id]
                                >= 1
                            )
                        )
                        # If the tasks use the same resource, then they must not
                        # overlap. We use a helper binary variable to capture the
                        # two cases:
                        # 1) t1 completes before t2
                        # 2) t2 completes before t1
                        self._model.addConstr(
                            (flag_var == 0)
                            >> (
                                self._task_ids_to_start_time[t1_id]
                                - self._task_ids_to_start_time[t2_id]
                                - task2.remaining_time
                                >= max_deadline * or_var
                            )
                        )
                        self._model.addConstr(
                            (flag_var == 0)
                            >> (
                                self._task_ids_to_start_time[t2_id]
                                - self._task_ids_to_start_time[t1_id]
                                - task1.remaining_time
                                >= max_deadline * (1 - or_var)
                            )
                        )

    def _add_objective(self):
        def sum_gains(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        for task in self._task_ids_to_task.values():
            # If the task is scheduled, then the gain is defined as slack
            # (i.e., deadline - finish time). Otherwise, set the gain to a very
            # small value.
            start_time = self._task_ids_to_start_time[task.id]
            self._model.addConstr(
                (self._task_ids_to_scheduled_flag[task.id] == 0)
                >> (
                    task.deadline - start_time - task.remaining_time
                    == self._task_ids_to_gain[task.id]
                )
            )
            self._model.addConstr(
                (self._task_ids_to_scheduled_flag[task.id] == 1)
                >> (
                    MIN_TASK_GAIN + task.deadline - start_time - task.remaining_time
                    == self._task_ids_to_gain[task.id]
                )
            )

        self._model.setObjective(
            sum_gains(self._task_ids_to_gain.values()), gp.GRB.MAXIMIZE
        )

    def _get_task_placement(self, task):
        res_index = round(self._task_ids_to_resources[task.id][0].X)
        if res_index == -1:
            # The task has been left unscheduled.
            placement = None
        else:
            placement = self._res_index_to_wp_id[res_index]
        return placement


class GurobiScheduler2(GurobiBaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        goal: str = "max_slack",
        enforce_deadlines: bool = True,
        lookahead: int = 0,
        _flags: Optional["absl.flags"] = None,
    ):
        super(GurobiScheduler2, self).__init__(
            preemptive, runtime, goal, enforce_deadlines, lookahead, _flags
        )
        # Mapping from task id to the var storing the placement.
        self._task_ids_to_placements = defaultdict(list)
        # Mapping from worker pool index to binary variables, each corresponding to a
        # task. The variable is set to one if the task is allocated to the work pool.
        self._wp_index_to_vars = defaultdict(list)

    def _initialize_state(
        self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools
    ):
        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.lookahead, self.preemptive, worker_pools
        )
        self._time = sim_time
        # Reset the state.
        self._task_ids_to_task = {}
        for task in tasks:
            self._task_ids_to_task[task.id] = task
        self._task_ids_to_start_time = {}
        self._task_ids_to_placements = defaultdict(list)
        self._wp_index_to_vars = defaultdict(list)
        self._task_graph = task_graph
        self._worker_pools = worker_pools
        self._model = gp.Model("RAP")
        self._model.Params.LogToConsole = 0
        self._model.Params.OptimalityTol = 0.0005
        # If the tolerance is too high, then the solver might pick an incorrect
        # solution.
        self._model.Params.IntFeasTol = 0.00001
        # self._model.Params.TimeLimit = 1  # In seconds.
        # Sets the solver method to concurrent and deterministic.
        # self._model.Params.Method = 4

    def _add_variables(self):
        for task in self._task_ids_to_task.values():
            # Add a variable to store the start time of the task.
            self._task_ids_to_start_time[task.id] = self._model.addVar(
                vtype=gp.GRB.INTEGER,
                lb=max(
                    self._time, task.release_time
                ),  # Start times cannot be less than sim time.
                ub=self._time + self.lookahead,
                name=f"start_time_{task.unique_name}",
            )
            # Add a variable which encodes if the task is scheduled or not.
            self._task_ids_to_placements[task.id] = [
                self._model.addVar(
                    vtype=gp.GRB.BINARY, name=f"no_placement_{task.unique_name}"
                )
            ]
            w_index = 1
            for wp in self._worker_pools._wps:
                # Add a variable which is set to 1 if a task is placed on the
                # variable's associated worker pool.
                placement_var = self._model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"placement_{task.unique_name}_worker_{w_index}",
                )
                self._task_ids_to_placements[task.id].append(placement_var)
                self._wp_index_to_vars[w_index].append((task, placement_var))
                w_index += 1

    def _add_task_resource_constraints(self):
        for task_id, task in self._task_ids_to_task.items():
            # The task can be left unscheduled or placed on a single worker.
            self._model.addConstr(
                gp.quicksum(p for p in self._task_ids_to_placements[task_id]) == 1
            )

        for task_id in self._task_ids_to_start_time.keys():
            # At each task start time find all the running tasks.
            task_event = self._task_ids_to_task[task_id]
            event_time = self._task_ids_to_start_time[task_id]
            # event_time = self._task_ids_to_task[task_id].release_time
            overlap_vars = {}
            for task in self._task_ids_to_task.values():
                start_time = self._task_ids_to_start_time[task.id]
                if task.id != task_id:
                    overlap_vars[task.id] = self._in_interval(
                        event_time,
                        start_time,
                        start_time + task.remaining_time,
                        var_name=f"overlap_{task_event.unique_name}_with_"
                        f"{task.unique_name}",
                    )
                    # overlap_vars[task.id] = self._in_interval_approximate(
                    #     event_time, task.release_time, task.deadline
                    # )
                else:
                    overlap_vars[task.id] = 1
            for res, quantity in task_event.resource_requirements.resources:
                if quantity == 0:
                    continue
                resource = Resource(name=res.name, _id="any")
                w_index = 1
                for wp in self._worker_pools._wps:
                    # Place constraints to ensure that worker pool's resources are not
                    # exceeded.
                    available = wp.resources.get_available_quantity(resource)
                    self._model.addConstr(
                        gp.quicksum(
                            worker_var
                            * overlap_vars[task.id]
                            * task.resource_requirements.get_available_quantity(
                                resource
                            )
                            for (task, worker_var) in self._wp_index_to_vars[w_index]
                        )
                        <= available,
                        name=f"time_{task_event.unique_name}_worker_{w_index}_"
                        f"res_{res.name}",
                    )
                    w_index += 1

    def _add_objective(self):
        objective = gp.QuadExpr()
        for task in self._task_ids_to_task.values():
            skipped = self._task_ids_to_placements[task.id][0]
            start_time = self._task_ids_to_start_time[task.id]
            if self._goal == "max_slack":
                objective.add(
                    (1 - skipped) * (task.deadline - task.remaining_time - start_time)
                )
                objective.add(
                    skipped
                    * (MIN_TASK_GAIN + task.deadline - task.remaining_time - start_time)
                )
            elif self._goal == "min_placement_delay":
                objective.add((1 - skipped) * (task.release_time - start_time))
                objective.add(
                    skipped * (MIN_TASK_GAIN + (task.release_time - start_time))
                )
            else:
                raise ValueError("Goal {self._goal} not supported.")
        self._model.setObjective(objective, gp.GRB.MAXIMIZE)

    def _get_task_placement(self, task):
        placement_vars = self._task_ids_to_placements[task.id]
        unscheduled = round(placement_vars[0].X)
        placement = None
        if unscheduled == 0:
            w_index = 1
            for wp in self._worker_pools._wps:
                scheduled = round(placement_vars[w_index].X)
                if scheduled:
                    placement = wp.id
                    break
                w_index += 1
        return placement
