import functools
import time

from typing import List, Optional, Sequence, Tuple

import gurobipy as gp

import utils
from schedulers.ilp_scheduler import ILPScheduler
from workload import Task, TaskGraph, TaskState
from workers import WorkerPool


class GurobiScheduler(ILPScheduler):

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
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

    def schedule(self, resource_requirements: List[List[bool]],
                 tasks: List[Task], dependency_matrix,
                 num_resources: List[int]):
        """Runs scheduling using Gurobi.

        Args:
            resource_requirements: List[List[bool]], List of lists of booleans,
                where each sublist is a list of booleans indicating whether a
                task conforms to a resource type.
            dependency_matrix (`List[List[bool]]`): List of lists of booleans,
                one for each task, indicating whether task i must finish
                before task j starts.
            num_resources (`List[int]`): Number of resources.
        """
        M = max([task.deadline for task in tasks])

        num_resources_ub = [
            sum(num_resources[0:i + 1]) for i in range(len(num_resources))
        ]
        num_resources_lb = [0] + [
            sum(num_resources[0:i + 1]) for i in range(len(num_resources) - 1)
        ]

        def safe_resource_index(resource_requirement: List[bool]):
            # check that there's only one True
            compats = [r for r in resource_requirement if r]
            if len(compats) > 1:
                raise Exception("More than one compatible resource")
            if len(compats) == 0:
                raise Exception("No compatible resource")
            return resource_requirement.index(True)

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        start_time = time.time()

        s = gp.Model('RAP')
        s.setParam("OptimalityTol", 1e-3)

        # We are solving for times and placements while minimizing costs.
        # Time when execution starts.
        times = [
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

        for time_var, task, cost_var in zip(times, tasks, costs):
            if self._enforce_deadlines:
                s.addConstr(time_var + task.runtime <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= time_var)
            # Defines cost as slack deadline - finish time.
            s.addConstr(task.deadline - time_var - task.runtime == cost_var)

        # Add constraints whether a task is placed on GPU or CPU.
        for p, res_req in zip(placements, resource_requirements):
            resource_index = safe_resource_index(
                res_req)  # convert bool vec to int index
            s.addConstr(p >= num_resources_lb[resource_index])
            s.addConstr(p <= num_resources_ub[resource_index] -
                        1)  # p < num_resources[resource_index]

        decision_vars = []
        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):

                # Dependent jobs need to finish before the next one.
                if dependency_matrix[row_i][col_j]:
                    s.addConstr(
                        times[row_i] + tasks[row_i].runtime <= times[col_j])
                if row_i < col_j:
                    # If two tasks are on the same resources, they must not
                    # overlap.
                    alpha = s.addVar(vtype=gp.GRB.BINARY,
                                     name=f'alpha{row_i}_{col_j}')
                    beta = s.addVar(vtype=gp.GRB.BINARY,
                                    name=f'beta{row_i}_{col_j}')
                    decision_vars.append((alpha, beta))

                    # pi > pj
                    s.addConstr(-alpha * M - beta * M +
                                1 <= placements[row_i] - placements[col_j])

                    # pj > pi
                    s.addConstr(alpha * M + (1 - beta) * M -
                                1 >= placements[row_i] - placements[col_j])

                    # tj - ti >= ei
                    s.addConstr(
                        times[col_j] - times[row_i] >= tasks[row_i].runtime -
                        (1 - alpha) * M - beta * M)

                    # ti - tj <= ej
                    s.addConstr(
                        times[row_i] - times[col_j] >= tasks[col_j].runtime -
                        (1 - alpha) * M - (1 - beta) * M)

        if not self._preemptive:
            # Add constraints for the running tasks.
            for i, task in enumerate(tasks):
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(placements[i] == 0)

        s.setObjective(MySum(costs), gp.GRB.MAXIMIZE)
        s.optimize()
        end_time = time.time()
        runtime = end_time - start_time

        if s.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {s.objVal}")
            assignment = [int(t.X)
                          for t in times], [int(p.X) for p in placements]
            return assignment, int(s.objVal), runtime
        elif s.status == gp.GRB.INFEASIBLE:
            self._logger.debug("No solution to solver run")
            return (None, None), None, runtime
        else:
            self._logger.debug(f"Opt solver end with status: {s.status}")
            return (None, None), None, runtime
