import functools
import pickle
import time
import sys

from typing import Optional, Sequence

import gurobipy as gp

import utils
from schedulers import BaseScheduler
from workload import Resource, Resources, Task, TaskGraph, TaskState
from workers import WorkerPools


class GurobiScheduler(BaseScheduler):

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 goal: str = 'max_slack',
                 enforce_deadlines: bool = True,
                 _flags: Optional['absl.flags'] = None):
        """Constructs a Gurobi scheduler.

        Args:
            goal (`str`): Goal of the scheduler run. Note: Gurobi does not
                support feasibility checking.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                schedule will return None.
        """
        self._preemptive = preemptive
        self._runtime = runtime
        assert goal != 'feasibility', \
            'Gurobi does not support feasibility checking.'
        self._goal = goal
        self._enforce_deadlines = enforce_deadlines
        self._time = None
        self._released_tasks = None
        self._task_graph = None
        self._worker_pools = None
        self._placements = None
        self._cost = None
        # Set up the loggers.
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
            self._scheduler_logger = utils.setup_scheduler_logging(
                name=self.__class__.__name__,
                log_file=_flags.scheduler_log_file_name)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)
            self._scheduler_logger = None

    def _add_task_dependency_constraints(self, s, tasks, dependency_matrix,
                                         times, placements):
        max_deadline = max([task.deadline for task in tasks])
        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                # Dependent jobs need to finish before the next one.
                if dependency_matrix[row_i][col_j]:
                    s.addConstr(
                        times[row_i] + tasks[row_i].runtime <= times[col_j])
                if row_i < col_j:
                    # Tasks using the same resources must not overlap.
                    alpha = s.addVar(vtype=gp.GRB.BINARY,
                                     name=f'alpha{row_i}_{col_j}')
                    beta = s.addVar(vtype=gp.GRB.BINARY,
                                    name=f'beta{row_i}_{col_j}')
                    # pi > pj
                    s.addConstr(-alpha * max_deadline - beta * max_deadline +
                                1 <= placements[row_i] - placements[col_j])
                    # pj > pi
                    s.addConstr(alpha * max_deadline +
                                (1 - beta) * max_deadline -
                                1 >= placements[row_i] - placements[col_j])
                    # tj - ti >= ei
                    s.addConstr(
                        times[col_j] - times[row_i] >= tasks[row_i].runtime -
                        (1 - alpha) * max_deadline - beta * max_deadline)
                    # ti - tj <= ej
                    s.addConstr(
                        times[row_i] - times[col_j] >= tasks[col_j].runtime -
                        (1 - alpha) * max_deadline - (1 - beta) * max_deadline)

    def _add_task_pinning_constraints(self, s, tasks, placements):
        if not self._preemptive:
            for i, task in enumerate(tasks):
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(placements[i] == 0)

    def _add_task_resource_constraints(self, s, tasks, res_type_to_id_range,
                                       placements):
        # Add constraints whether a task is placed on GPU or CPU.
        # from num_cpus to num_cpus to num_gpus.
        for placement, task in zip(placements, tasks):
            for r_name, (start_id, end_id) in res_type_to_id_range.items():
                unit_resource = Resources(
                    {Resource(name=r_name, _id="any"): 1})
                if task.resource_requirements >= unit_resource:
                    s.addConstr(placement >= start_id)
                    s.addConstr(placement <= end_id - 1)

    def _get_schedulable_tasks(self, released_tasks, wps):
        # Create the tasks to be scheduled.
        if self.preemptive:
            # Collect all the currently placed tasks on the WorkerPools, along
            # with the set of released tasks.
            tasks = [task for task in released_tasks]
            tasks.extend(wps.get_placed_tasks())
        else:
            # Collect the currently released tasks.
            tasks = [task for task in released_tasks]
        return tasks

    def schedule(self, sim_time: float, released_tasks: Sequence[Task],
                 task_graph: TaskGraph, wps: WorkerPools):

        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        self._time = sim_time
        self._released_tasks = released_tasks
        self._task_graph = task_graph
        self._worker_pools = wps
        # TODO: We should get tasks that will be released later as well.
        tasks = self._get_schedulable_tasks(released_tasks, wps)

        scheduler_start_time = time.time()
        s = gp.Model('RAP')
        s.setParam("OptimalityTol", 1e-3)

        # We are solving for start_times and placements while minimizing costs.
        start_times = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f't{i}')
            for i in range(0, len(tasks))
        ]
        # Costs of gap.
        costs = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f'c{i}')
            for i in range(0, len(tasks))
        ]
        # Placement on CPU or GPU.
        placements = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f'p{i}')
            for i in range(0, len(tasks))
        ]

        for start_time, task, cost_var in zip(start_times, tasks, costs):
            if self._enforce_deadlines:
                s.addConstr(start_time + task.runtime <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.addConstr(task.deadline - start_time - task.runtime == cost_var)

        (res_type_to_id_range,
         res_id_to_wp_id) = wps.get_resource_ilp_encoding()
        self._add_task_resource_constraints(s, tasks, res_type_to_id_range,
                                            placements)
        # TODO: This doesn't account for the task dependencies.
        dependency_matrix = [[False] * len(tasks)] * len(tasks)
        self._add_task_dependency_constraints(s, tasks, dependency_matrix,
                                              start_times, placements)
        self._add_task_pinning_constraints(s, tasks, placements)

        s.setObjective(sum_costs(costs), gp.GRB.MAXIMIZE)
        s.optimize()
        scheduler_end_time = time.time()
        self._runtime = scheduler_end_time - scheduler_start_time\
            if self.runtime == -1 else self.runtime

        if s.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {s.objVal}")
            self._verify_schedule(tasks, dependency_matrix, start_times,
                                  placements)
            self._placements = []
            for task, start_time, placement in zip(tasks, start_times,
                                                   placements):
                if int(start_time.X) <= sim_time + self.runtime:
                    # TODO: It only places tasks with a start time smaller
                    # than the time at the end of the scheduler run.
                    self._placements.append(
                        (task, res_id_to_wp_id[int(placement.X)]))
                else:
                    self._placements.append((task, None))
            self._cost = int(s.objVal)
            return self.runtime, self._placements
        else:
            self._placements = [(task, None) for task in tasks]
            self._cost = sys.maxsize
            if s.status == gp.GRB.INFEASIBLE:
                # TODO: Implement load shedding.
                self._logger.debug("Solver couldn't find a solution.")
                return self.runtime, self._placements
            else:
                self._logger.debug(f"Solver failed with status: {s.status}")
                return self.runtime, self._placements

    def log(self):
        if self._flags.scheduler_log_file_name is not None:
            with open(self._flags.scheduler_log_file_name, 'wb') as log_file:
                logged_data = {
                    'time': self._time,
                    'released_tasks': self._released_tasks,
                    'task_graph': self._task_graph,
                    'worker_pools': self._worker_pools,
                    'runtime': self.runtime,
                    'placements': self._placementsresult,
                    'cost': self._cost
                }
                pickle.dump(logged_data, log_file)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime
