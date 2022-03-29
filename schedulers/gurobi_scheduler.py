import functools
import pickle
import sys
import time
from typing import Optional

import absl  # noqa: F401
import gurobipy as gp

import utils
from schedulers import BaseScheduler
from workers import WorkerPools
from workload import TaskGraph, TaskState


class GurobiScheduler(BaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        goal: str = "max_slack",
        enforce_deadlines: bool = True,
        scheduling_horizon: int = 0,
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
            scheduling_horizon (`int`): The scheduler will try to place
                tasks that are within the scheduling horizon (in us) using
                estimated task release times.
        """
        self._preemptive = preemptive
        self._runtime = runtime
        assert goal != "feasibility", "Gurobi does not support feasibility checking."
        self._goal = goal
        self._enforce_deadlines = enforce_deadlines
        self._scheduling_horizon = scheduling_horizon
        self._time = None
        self._task_ids_to_task = {}
        # Mapping from task id to the var storing the task start time.
        self._task_ids_to_start_time = {}
        # Mapping from task id to the var storing the task placement.
        self._task_ids_to_placement = {}
        # Mapping from task id to the var storing the cost of the placement.
        self._task_ids_to_cost = {}
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

    def _add_task_timing_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                s.addConstr(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.addConstr(
                task.deadline - start_time - task.remaining_time
                == self._task_ids_to_cost[task_id]
            )

    def _add_task_dependency_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            children = self._task_graph.get_children(task)
            for child_task in children:
                # Dependent tasks need to finish before the next one.
                if child_task.id in self._task_ids_to_start_time:
                    s.addConstr(
                        self._task_ids_to_start_time[task_id] + task.remaining_time
                        <= self._task_ids_to_start_time[child_task.id]
                    )

    def _add_task_pinning_constraints(self, s):
        if not self._preemptive:
            for task_id, task in self._task_ids_to_task.items():
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(self._task_ids_to_placement[task_id] == 0)

    def _add_task_resource_constraints(self, s, res_type_to_id_range):
        # Add constraints whether a task is placed on GPU or CPU.
        # from num_cpus to num_cpus to num_gpus.
        for task_id, task in self._task_ids_to_task.items():
            if len(task.resource_requirements) > 1:
                self._logger.error(
                    "Scheduler doesn't support multi-resource requirements"
                )
            # for res in task.resource_requirements._resource_vector.keys():
            # TODO: Add constraints for all resources once multi-dimensional
            # requirements are supported.
            resource = next(iter(task.resource_requirements._resource_vector))
            (start_id, end_id) = res_type_to_id_range[resource.name]
            s.addConstr(self._task_ids_to_placement[task_id] >= start_id)
            s.addConstr(self._task_ids_to_placement[task_id] <= end_id - 1)

        # Tasks using the same resources must not overlap.
        if len(self._task_ids_to_task) == 0:
            return
        max_deadline = max([task.deadline for task in self._task_ids_to_task.values()])
        for t1_id, task1 in self._task_ids_to_task.items():
            for t2_id, task2 in self._task_ids_to_task.items():
                if t2_id >= t1_id:
                    continue
                alpha = s.addVar(vtype=gp.GRB.BINARY, name=f"alpha{t1_id}_{t2_id}")
                betas = s.addVars(2, vtype=gp.GRB.BINARY, name=f"beta{t1_id}_{t2_id}")
                # Helper variable which is 0 if the tasks have been placed on the same
                # resource.
                s.addConstr(
                    alpha
                    == self._task_ids_to_placement[t1_id]
                    - self._task_ids_to_placement[t2_id]
                )
                # If the tasks use the same resource, then they must not overlap.
                # We uses a helper binary variable to capture the two cases:
                # 1) t1 completes before t2
                # 2) t2 completes before t1
                s.addConstr(betas[0] + betas[1] == 1)
                s.addConstr(
                    (alpha == 0)
                    >> (
                        self._task_ids_to_start_time[t2_id]
                        - self._task_ids_to_start_time[t1_id]
                        - task1.remaining_time
                        >= 0 - max_deadline * (1 - betas[0])
                    )
                )
                s.addConstr(
                    (alpha == 0)
                    >> (
                        self._task_ids_to_start_time[t1_id]
                        - self._task_ids_to_start_time[t2_id]
                        - task2.remaining_time
                        >= 0 - max_deadline * (1 - betas[1])
                    )
                )

    def schedule(self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools):
        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        self._time = sim_time
        # Rest the state.
        self._task_ids_to_task = {}
        self._task_ids_to_start_time = {}
        self._task_ids_to_placement = {}
        self._task_ids_to_cost = {}
        self._task_graph = task_graph
        self._worker_pools = worker_pools

        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.scheduling_horizon, self.preemptive, worker_pools
        )

        scheduler_start_time = time.time()
        s = gp.Model("RAP")
        s.setParam("OptimalityTol", 1e-3)

        # We are solving for start_times and placements while minimizing costs.
        for task in tasks:
            self._task_ids_to_task[task.id] = task
            self._task_ids_to_start_time[task.id] = s.addVar(
                vtype=gp.GRB.INTEGER, name=f"t{task.id}"
            )
            self._task_ids_to_cost[task.id] = s.addVar(
                vtype=gp.GRB.INTEGER, name=f"c{task.id}"
            )
            self._task_ids_to_placement[task.id] = s.addVar(
                vtype=gp.GRB.INTEGER, name=f"p{task.id}"
            )

        (
            res_type_to_id_range,
            res_id_to_wp_id,
        ) = worker_pools.get_resource_ilp_encoding()

        self._add_task_timing_constraints(s)
        self._add_task_resource_constraints(s, res_type_to_id_range)
        self._add_task_dependency_constraints(s)
        self._add_task_pinning_constraints(s)

        s.setObjective(sum_costs(self._task_ids_to_cost.values()), gp.GRB.MAXIMIZE)
        s.optimize()
        scheduler_end_time = time.time()
        self._runtime = (
            int((scheduler_end_time - scheduler_start_time) * 1000000)
            if self.runtime == -1
            else self.runtime
        )

        if s.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {s.objVal}")
            self._placements = []
            self._cost = int(s.objVal)
            for task_id, task in self._task_ids_to_task.items():
                start_time = int(self._task_ids_to_start_time[task_id].X)
                placement = res_id_to_wp_id[int(self._task_ids_to_placement[task_id].X)]
                if start_time <= sim_time + self.runtime * 2:
                    # We only place the tasks with a start time earlier than
                    # the estimated end time of the next scheduler run.
                    # Therefore, a task can progress before the next scheduler
                    # finishes. However, the next scheduler will assume that
                    # the task is not running while considering for placement.
                    self._placements.append((task, placement))
                else:
                    self._placements.append((task, None))
            start_times = {}
            for task_id, st_var in self._task_ids_to_start_time.items():
                start_times[task_id] = int(st_var.X)
            self._verify_schedule(
                self._worker_pools, self._task_graph, self._placements, start_times
            )
        else:
            self._placements = [
                (task, None) for task in self._task_ids_to_task.values()
            ]
            self._cost = sys.maxsize
            if s.status == gp.GRB.INFEASIBLE:
                # TODO: Implement load shedding.
                self._logger.debug("Solver couldn't find a solution.")
            else:
                self._logger.debug(f"Solver failed with status: {s.status}")
        # Log the scheduler run.
        self.log()
        return self.runtime, self._placements

    def log(self):
        if self._flags is not None and self._flags.scheduler_log_file_name is not None:
            with open(self._flags.scheduler_log_file_name, "wb") as log_file:
                logged_data = {
                    "time": self._time,
                    "tasks": self._task_ids_to_task,
                    "task_graph": self._task_graph,
                    "worker_pools": self._worker_pools,
                    "scheduling_horizon": self._scheduling_horizon,
                    "runtime": self.runtime,
                    "placements": self._placements,
                    "cost": self._cost,
                }
                pickle.dump(logged_data, log_file)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime

    @property
    def scheduling_horizon(self):
        return self._scheduling_horizon
