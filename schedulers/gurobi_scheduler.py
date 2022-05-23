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

# Minimum task cost constant, which is used to set the cost of leaving a task
# unscheduled. It is equivalent to missing a task deadline by 100 seconds.
MIN_TASK_COST = -100 * 1000 * 1000


class GurobiScheduler(BaseScheduler):
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
            goal (`str`): Goal of the scheduler run. Note: Gurobi does not
                support feasibility checking.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                schedule will return None.
            lookahead (`int`): The scheduler will try to place tasks that are within
                the scheduling lookahead (in us) using estimated task release times.
        """
        super(GurobiScheduler, self).__init__(
            preemptive, runtime, lookahead, enforce_deadlines, _flags
        )
        assert goal != "feasibility", "Gurobi does not support feasibility checking."
        self._goal = goal
        self._time = None
        self._task_ids_to_task = {}
        # Mapping from task id to the var storing the task start time.
        self._task_ids_to_start_time = {}
        # Mapping from task id to the var storing the task resource placement.
        self._task_ids_to_resources = defaultdict(list)
        # Mapping from task id to the variable storing the cost of the placement.
        self._task_ids_to_cost = {}
        # Mapping from task id to the variable controlling if the task is placed
        # or not.
        self._task_ids_to_scheduled_flag = {}
        self._task_graph = None
        self._worker_pools = None
        self._placements = []

    def _add_variables(
        self,
        sim_time,
        s,
        tasks,
        num_workers,
        res_type_index_range,
        res_index_to_wp_index,
    ):
        num_resources = len(res_index_to_wp_index.keys())
        # We are solving for start_times and placements while minimizing costs.
        for task in tasks:
            self._task_ids_to_task[task.id] = task
            # Add a variable to store the cost of the task assignment.
            self._task_ids_to_cost[task.id] = s.addVar(
                vtype=gp.GRB.INTEGER,
                lb=MIN_TASK_COST,
                ub=task.deadline
                - sim_time
                - task.remaining_time,  # Task cannot start before sim time.
                name=f"cost_task_{task.id}",
            )
            # Add a variable to store the start time of the task.
            self._task_ids_to_start_time[task.id] = s.addVar(
                vtype=gp.GRB.INTEGER,
                lb=sim_time,  # Start times cannot be less than sim time.
                name=f"start_time_task_{task.id}",
            )
            # Add a variable to control if the task is placed or left unplaced.
            # A value of 0 means that the task is be placed, and a value of 1
            # means that the task is not placed.
            # Note: Do not change the encoding as the variable is also used to
            # ensure that two tasks do not overlap on the same resource.
            scheduled_flag_var = s.addVar(
                vtype=gp.GRB.BINARY, name=f"scheduled_task_{task.id}"
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
                    res_var = s.addVar(
                        vtype=gp.GRB.INTEGER,
                        lb=-1,
                        ub=num_resources - 1,
                        name=f"resource_task_{task.id}_{resource.name}_{index}",
                    )
                    # Stores the id of worker containing the resource that got
                    # allocated to the task.
                    worker_var = s.addVar(
                        vtype=gp.GRB.INTEGER,
                        lb=-1,
                        ub=num_workers - 1,
                        name=f"worker_task_{task.id}_{resource.name}_{index}",
                    )
                    self._task_ids_to_resources[task.id].append(res_var)
                    # Add constraints whether a task is placed on GPU or CPU.
                    (start_index, end_index) = res_type_index_range[resource.name]
                    # If the task is not scheduled, then its resource and worker
                    # variables must be -1.
                    s.addConstr((scheduled_flag_var == 1) >> (res_var == -1))
                    s.addConstr((scheduled_flag_var == 1) >> (worker_var == -1))
                    s.addConstr((scheduled_flag_var == 0) >> (res_var >= start_index))
                    s.addConstr((scheduled_flag_var == 0) >> (res_var <= end_index - 1))
                    # if ri == resource_index => worker_var == worker_index, where
                    # worker_index is the index of the worker the resource in part of.
                    for res_index in range(start_index, end_index):
                        res_diff_var = s.addVar(
                            vtype=gp.GRB.INTEGER,
                            lb=-num_resources,
                            ub=num_resources,
                            name=f"resource_diff_task_{task.id}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        abs_diff_var = s.addVar(
                            lb=0,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"abs_resource_diff_task_{task.id}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        flag_var = s.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"flag_resource_diff_task_{task.id}_"
                            f"{resource.name}_{index}_{res_index}",
                        )
                        s.addConstr(res_diff_var == res_var - res_index)
                        s.addConstr(abs_diff_var == gp.abs_(res_diff_var))
                        s.addConstr((flag_var == 1) >> (abs_diff_var >= 1))
                        # If the resource var if equal to the index then its
                        # corresponding worker variable must equal to the worker index
                        # of the resource.
                        s.addConstr(
                            (flag_var == 0)
                            >> (worker_var == res_index_to_wp_index[res_index])
                        )
                        s.addConstr((flag_var == 0) >> (abs_diff_var == 0))

                    # A task must receive all its resources on a single worker.
                    if prev_worker_var:
                        s.addConstr(prev_worker_var == worker_var)
                    prev_worker_var = worker_var

    def _add_task_timing_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                s.addConstr(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= start_time)
            # If the task is scheduled, then the cost is defined as slack
            # (i.e., deadline - finish time). Otherwise, set the cost to a very
            # small value.
            s.addConstr(
                (self._task_ids_to_scheduled_flag[task_id] == 0)
                >> (
                    task.deadline - start_time - task.remaining_time
                    == self._task_ids_to_cost[task_id]
                )
            )
            s.addConstr(
                (self._task_ids_to_scheduled_flag[task_id] == 1)
                >> (
                    MIN_TASK_COST + task.deadline - start_time - task.remaining_time
                    == self._task_ids_to_cost[task_id]
                )
            )

    def _add_task_dependency_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            for child_task in self._task_graph.get_children(task):
                # Dependent tasks need to finish before the next one.
                if child_task.id in self._task_ids_to_start_time:
                    s.addConstr(
                        self._task_ids_to_start_time[task_id] + task.remaining_time
                        <= self._task_ids_to_start_time[child_task.id]
                    )

    def _add_task_resource_constraints(self, s, num_resources):
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
                        diff_var = s.addVar(
                            lb=-num_resources,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"res_diff_task_{t1_id}_res_index_{r1_index}_"
                            f"task_{t2_id}_res_index_{r2_index}",
                        )
                        abs_diff_var = s.addVar(
                            lb=0,
                            ub=num_resources,
                            vtype=gp.GRB.INTEGER,
                            name=f"abs_res_diff_task_{t1_id}_res_index_{r1_index}_"
                            f"task_{t2_id}_res_index_{r2_index}",
                        )
                        flag_var = s.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"flag_task_{t1_id}_res_index_{r1_index}_task_"
                            f"{t2_id}_res_index_{r2_index}",
                        )
                        or_var = s.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"or_task_{t1_id}_res_index_{r1_index}_task_"
                            f"{t2_id}_res_index_{r2_index}",
                        )
                        s.addConstr(diff_var == t1_res - t2_res)
                        s.addConstr(abs_diff_var == gp.abs_(diff_var))
                        # Ensure that either the tasks do not use the same resource,
                        # or at least one of them was left unscheduled.
                        s.addConstr(
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
                        s.addConstr(
                            (flag_var == 0)
                            >> (
                                self._task_ids_to_start_time[t1_id]
                                - self._task_ids_to_start_time[t2_id]
                                - task2.remaining_time
                                >= max_deadline * or_var
                            )
                        )
                        s.addConstr(
                            (flag_var == 0)
                            >> (
                                self._task_ids_to_start_time[t2_id]
                                - self._task_ids_to_start_time[t1_id]
                                - task1.remaining_time
                                >= max_deadline * (1 - or_var)
                            )
                        )

    def schedule(self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools):
        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        self._time = sim_time
        # Rest the state.
        self._task_ids_to_task = {}
        self._task_ids_to_start_time = {}
        self._task_ids_to_resources = defaultdict(list)
        self._task_ids_to_cost = {}
        self._task_graph = task_graph
        self._worker_pools = worker_pools

        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.lookahead, self.preemptive, worker_pools
        )

        scheduler_start_time = time.time()
        s = gp.Model("RAP")
        s.Params.OptimalityTol = 0.005
        s.Params.IntFeasTol = 0.01
        # s.Params.TimeLimit = 1  # In seconds.
        # Sets the solver method to concurrent and deterministic.
        # s.Params.Method = 4

        (
            res_type_index_range,
            res_index_to_wp_id,
            res_index_to_wp_index,
            num_workers,
        ) = worker_pools.get_resource_ilp_encoding()

        # We are solving for start_times and placements while minimizing costs.
        self._add_variables(
            sim_time, s, tasks, num_workers, res_type_index_range, res_index_to_wp_index
        )
        self._add_task_timing_constraints(s)
        self._add_task_resource_constraints(s, len(res_index_to_wp_index.keys()))
        self._add_task_dependency_constraints(s)

        s.setObjective(sum_costs(self._task_ids_to_cost.values()), gp.GRB.MAXIMIZE)
        s.optimize()
        scheduler_end_time = time.time()
        runtime = int((scheduler_end_time - scheduler_start_time) * 1000000)
        self._logger.info(f"Scheduler wall-clock runtime: {runtime}")
        if self.runtime != -1:
            runtime = self.runtime

        if s.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {s.objVal}")
            self._placements = []
            self._cost = int(s.objVal)
            for task_id, task in self._task_ids_to_task.items():
                # All the variables of a task were placed on the same WorkerPool
                # and at the same time. Thus, we can extract the placement from the
                # first variable.
                start_time = int(self._task_ids_to_start_time[task_id].X)
                res_index = int(self._task_ids_to_resources[task_id][0].X)
                if res_index == -1:
                    # The task has been left unscheduled.
                    placement = None
                else:
                    placement = res_index_to_wp_id[res_index]
                if start_time <= sim_time + runtime * 2:
                    # We only place the tasks with a start time earlier than
                    # the estimated end time of the next scheduler run.
                    # Therefore, a task can progress before the next scheduler
                    # finishes. However, the next scheduler will assume that
                    # the task is not running while considering for placement.
                    self._placements.append((task, placement, start_time))
                else:
                    self._placements.append((task, None, None))
            self._verify_schedule(
                self._worker_pools, self._task_graph, self._placements
            )
        else:
            self._placements = [
                (task, None, None) for task in self._task_ids_to_task.values()
            ]
            self._cost = sys.maxsize
            if s.status == gp.GRB.INFEASIBLE:
                self._logger.debug("Solver couldn't find a solution.")
                self.log_solver_internal(s)
            else:
                self._logger.debug(f"Solver failed with status: {s.status}")
                self.log_solver_internal(s)
        # Log the scheduler run.
        self.log()
        return runtime, self._placements

    def log_solver_internal(self, s):
        self._logger.debug(f"Solver stats: {s.printStats()}")
        s.computeIIS()
        self._logger.debug("The following constraint(s) cannot be satisfied:")
        for c in s.getConstrs():
            if c.IISConstr:
                self._logger.debug(f"Constraint: {c}")

    def log(self):
        if self._flags is not None and self._flags.scheduler_log_file_name is not None:
            with open(self._flags.scheduler_log_file_name, "wb") as log_file:
                logged_data = {
                    "time": self._time,
                    "tasks": self._task_ids_to_task,
                    "task_graph": self._task_graph,
                    "worker_pools": self._worker_pools,
                    "lookahead": self._lookahead,
                    "runtime": self.runtime,
                    "placements": self._placements,
                    "cost": self._cost,
                }
                pickle.dump(logged_data, log_file)


class GurobiScheduler2(BaseScheduler):
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
            goal (`str`): Goal of the scheduler run. Note: Gurobi does not
                support feasibility checking.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                schedule will return None.
            lookahead (`int`): The scheduler will try to place
                tasks that are within the scheduling lookahead (in us) using
                estimated task release times.
        """
        super(GurobiScheduler2, self).__init__(
            preemptive, runtime, lookahead, enforce_deadlines
        )
        assert goal != "feasibility", "Gurobi does not support feasibility checking."
        self._goal = goal
        self._time = None
        self._task_ids_to_task = {}
        # Mapping from task id to the var storing the task start time.
        self._task_ids_to_start_time = {}
        # Mapping from task id to the var storing the placement.
        self._task_ids_to_placements = defaultdict(list)
        # Mapping from worker pool index to binary variables, each corresponding to a
        # task. The variable is set to one if the task is allocated to the work pool.
        self._wp_index_to_vars = defaultdict(list)
        self._task_graph = None
        self._worker_pools = None
        self._placements = []
        # Set up the loggers.
        self._flags = _flags
        if _flags:
            self._logger = utils.setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

    def _add_variables(self, sim_time, model, tasks):
        for task in tasks:
            self._task_ids_to_task[task.id] = task
            # Add a variable to store the start time of the task.
            # TODO: Should we set an upper bound?
            self._task_ids_to_start_time[task.id] = model.addVar(
                vtype=gp.GRB.INTEGER,
                lb=max(
                    sim_time, task.release_time
                ),  # Start times cannot be less than sim time.
                name=f"start_time_task_{task.id}",
            )
            # Add a variable which encodes if the task is scheduled or not.
            self._task_ids_to_placements[task.id] = [
                model.addVar(vtype=gp.GRB.BINARY, name=f"no_placement_{task.id}")
            ]
            w_index = 1
            for wp in self._worker_pools.worker_pools:
                # Add a variable which is set to 1 if a task is placed on the
                # variable's associated worker pool.
                placement_var = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"placement_{task.id}_worker_{w_index}",
                )
                self._task_ids_to_placements[task.id].append(placement_var)
                self._wp_index_to_vars[w_index].append((task, placement_var))
                w_index += 1

    def _add_task_timing_constraints(self, model):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                model.addConstr(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            model.addConstr(task.release_time <= start_time)

    def _add_task_dependency_constraints(self, model):
        for task_id, task in self._task_ids_to_task.items():
            for child_task in self._task_graph.get_children(task):
                # Dependent tasks need to finish before the next one.
                if child_task.id in self._task_ids_to_start_time:
                    model.addConstr(
                        self._task_ids_to_start_time[task_id] + task.remaining_time
                        <= self._task_ids_to_start_time[child_task.id]
                    )

    def _in_interval(self, model, event_time, start_time, end_time, var_name=None):
        """Adds constraints to check if event time is in [start, end] time interval.

        Args:
            model: The Gurobi model.
            event_time: The time of the event to check if is in the interval.
            start_time: The start of the time interval.
            end_time: The end of the time interval.
            var_name: Name to give to the Gurobi binary variable.

        Returns:
            A Gurobi variable that is 1 if the time is in the interval, 0 otherwise.
        """
        if var_name is None:
            var_name = f"time_{event_time}_in_{start_time},{end_time}"
        in_var = model.addVar(vtype=gp.GRB.BINARY, name=var_name)
        M = 1000000
        model.addConstr(event_time >= start_time - M * (1 - in_var))
        model.addConstr(event_time <= start_time - 1 + M * in_var)
        model.addConstr(event_time <= end_time + M * (1 - in_var))
        model.addConstr(event_time >= end_time + 1 - M * in_var)
        return in_var

    def _in_interval_approximate(self, event_time, start_time, end_time):
        if start_time <= event_time and event_time <= end_time:
            return 1
        return 0

    def _add_task_resource_constraints(self, model):
        for task_id, task in self._task_ids_to_task.items():
            # The task can be left unscheduled or placed on a single worker.
            model.addConstr(
                gp.quicksum(p for p in self._task_ids_to_placements[task_id]) == 1
            )

        total_resources = Resources(_logger=self._logger)
        for wp in self._worker_pools.worker_pools:
            total_resources += wp.resources
        res_names = set(
            map(attrgetter("name"), total_resources._resource_vector.keys())
        )

        for task_id in self._task_ids_to_start_time.keys():
            event_time = self._task_ids_to_start_time[task_id]
            # event_time = self._task_ids_to_task[task_id].release_time
            overlap_vars = {}
            for task in self._task_ids_to_task.values():
                start_time = self._task_ids_to_start_time[task.id]
                overlap_vars[task.id] = self._in_interval(
                    model, event_time, start_time, start_time + task.remaining_time
                )
                # overlap_vars[task.id] = self._in_interval_approximate(
                #     event_time, task.release_time, task.deadline
                # )

            w_index = 1
            for wp in self._worker_pools.worker_pools:
                for res_name in res_names:
                    resource = Resource(name=res_name, _id="any")
                    # Place constraints to ensure that worker pool's resources are not
                    # exceeded.
                    available = wp.resources.get_available_quantity(resource)
                    model.addConstr(
                        gp.quicksum(
                            worker_var
                            * overlap_vars[task.id]
                            * task.resource_requirements.get_available_quantity(
                                resource
                            )
                            for (task, worker_var) in self._wp_index_to_vars[w_index]
                        )
                        <= available
                    )
                w_index += 1

    def schedule(self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools):
        self._time = sim_time
        # Reset the state.
        self._task_ids_to_task = {}
        self._task_ids_to_start_time = {}
        self._task_ids_to_placements = defaultdict(list)
        self._wp_index_to_vars = defaultdict(list)
        self._task_graph = task_graph
        self._worker_pools = worker_pools

        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.lookahead, self.preemptive, worker_pools
        )

        scheduler_start_time = time.time()
        model = gp.Model("RAP")
        model.Params.OptimalityTol = 0.005
        model.Params.IntFeasTol = 0.01
        # model.Params.TimeLimit = 1  # In seconds.
        # Sets the solver method to concurrent and deterministic.
        # model.Params.Method = 4

        self._add_variables(sim_time, model, tasks)
        self._add_task_timing_constraints(model)
        self._add_task_dependency_constraints(model)
        self._add_task_resource_constraints(model)

        objective = gp.QuadExpr()
        for task in tasks:
            skipped = self._task_ids_to_placements[task.id][0]
            start_time = self._task_ids_to_start_time[task.id]
            objective.add(
                (1 - skipped) * (task.deadline - task.remaining_time - start_time)
            )
            objective.add(
                skipped
                * (MIN_TASK_COST + task.deadline - task.remaining_time - start_time)
            )
        model.setObjective(objective, gp.GRB.MAXIMIZE)
        model.optimize()
        scheduler_end_time = time.time()
        runtime = int((scheduler_end_time - scheduler_start_time) * 1000000)
        self._logger.info(f"Scheduler wall-clock runtime: {runtime}")
        if self.runtime != -1:
            runtime = self.runtime

        if model.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {model.objVal}")
            self._placements = []
            self._cost = int(model.objVal)
            for task in tasks:
                start_time = int(self._task_ids_to_start_time[task.id].X)
                placement_vars = self._task_ids_to_placements[task.id]
                unscheduled = int(placement_vars[0].X)
                placement = None
                if unscheduled == 0:
                    w_index = 1
                    for wp in worker_pools.worker_pools:
                        scheduled = int(placement_vars[w_index].X)
                        if scheduled:
                            placement = wp.id
                            break
                        w_index += 1
                if start_time <= sim_time + runtime * 2:
                    # We only place the tasks with a start time earlier than
                    # the estimated end time of the next scheduler run.
                    # Therefore, a task can progress before the next scheduler
                    # finishes. However, the next scheduler will assume that
                    # the task is not running while considering for placement.
                    self._placements.append((task, placement, start_time))
                else:
                    self._placements.append((task, None, None))
            self._verify_schedule(worker_pools, self._task_graph, self._placements)
        else:
            self._placements = [(task, None, None) for task in tasks]
            self._cost = sys.maxsize
            if model.status == gp.GRB.INFEASIBLE:
                self._logger.debug("Solver couldn't find a solution.")
                self.log_solver_internal(model)
            else:
                self._logger.debug(f"Solver failed with status: {model.status}")
                self.log_solver_internal(model)
        # Log the scheduler run.
        self.log()
        return runtime, self._placements

    def log_solver_internal(self, model):
        self._logger.debug(f"Solver stats: {model.printStats()}")
        model.computeIIS()
        self._logger.debug("The following constraint(s) cannot be satisfied:")
        for c in model.getConstrs():
            if c.IISConstr:
                self._logger.debug(f"Constraint: {c}")

    def log(self):
        if self._flags is not None and self._flags.scheduler_log_file_name is not None:
            with open(self._flags.scheduler_log_file_name, "wb") as log_file:
                logged_data = {
                    "time": self._time,
                    "tasks": self._task_ids_to_task,
                    "task_graph": self._task_graph,
                    "worker_pools": self._worker_pools,
                    "lookahead": self._lookahead,
                    "runtime": self.runtime,
                    "placements": self._placements,
                    "cost": self._cost,
                }
                pickle.dump(logged_data, log_file)
