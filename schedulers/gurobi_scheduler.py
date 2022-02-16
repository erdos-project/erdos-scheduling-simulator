import functools
import logging
import time

from typing import Dict, List, Optional

import gurobipy as gp

import utils
from schedulers.ilp_scheduler import ILPScheduler


class GurobiScheduler(ILPScheduler):

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 _logger: Optional[logging.Logger] = None):
        self._preemptive = preemptive
        self._runtime = runtime
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=f"gurobi")

    def schedule(self,
                 needs_gpu: List[bool],
                 release_times: List[int],
                 absolute_deadlines: List[int],
                 expected_runtimes: List[int],
                 dependency_matrix,
                 pinned_tasks: List[int],
                 num_tasks: int,
                 num_resources: Dict[str, int],
                 bits=None,
                 goal='max_slack',
                 log_dir=None,
                 logger: Optional[logging.Logger] = None):
        """Runs scheduling using Gurobi.

        Args:
            needs_gpu (`List[bool]`): List of booleans, one for each task,
                indicating whether the task requires a GPU.
            release_times (`List[int]`): List of ints, one for each task,
                indicating the release time of the task.
            absolute_deadlines (`List[int]`): List of ints, one for each task,
                indicating the absolute deadline of the task.
            expected_runtimes (`List[int]`): List of ints, one for each task,
                indicating the expected runtime of the task.
            dependency_matrix (`List[List[bool]]`): List of lists of booleans,
                one for each task, indicating whether task i must finish
                before task j starts.
            pinned_tasks (`List[int]`): List of ints, one for each task,
                indicating the hardware index if a task is pinned to that
                resource (or already running there).
            num_tasks (`int`): Number of tasks.
            num_resources (`Dict[str,int]`): Number of resources.
            bits (`int`): Number of bits to use for the ILP.
            goal (`str`): Goal of the scheduler run. Note: Gurobi does not
                support feasibility checking.
            log_dir (`str`): Directory to write the ILP to.
            logger(`Optional[logging.Logger]`): The logger to use to log the
                results of the execution.
        """
        assert goal != 'feasibility', \
            'Gurobi does not support feasibility checking.'
        M = max(absolute_deadlines)
        num_gpus = num_resources['GPU']
        num_cpus = num_resources['CPU']

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        start_time = time.time()

        s = gp.Model('RAP')
        s.setParam("OptimalityTol", 1e-3)

        # We are solving for times and placements while minimizing costs.
        times = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f't{i}')
            for i in range(0, num_tasks)
        ]  # Time when execution starts.
        costs = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f'c{i}')
            for i in range(0, num_tasks)
        ]  # Costs of gap.
        placements = [
            s.addVar(vtype=gp.GRB.INTEGER, name=f'p{i}')
            for i in range(0, num_tasks)
        ]  # Placement on CPU or GPU.

        for t, r, e, d, c in zip(times, release_times, expected_runtimes,
                                 absolute_deadlines, costs):
            # Finish before deadline.
            s.addConstr(t + e <= d - 1)
            # Start at or after release time.
            s.addConstr(r <= t)
            # Defines cost as slack deadline - finish time.
            s.addConstr(d - t - e == c)

        # Add constraints whether a task is placed on GPU or CPU.
        for p, gpu in zip(placements, needs_gpu):
            if gpu:
                s.addConstr(p >= num_cpus + 1)
                s.addConstr(p <=
                            (num_gpus + num_cpus))  # second half is num_gpus
            else:
                s.addConstr(p >= 1)
                s.addConstr(p <= num_cpus)  # first half is num_cpus

        decision_vars = []
        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):

                # Dependent jobs need to finish before the next one.
                if dependency_matrix[row_i][col_j]:
                    s.addConstr(times[row_i] +
                                expected_runtimes[row_i] <= times[col_j] - 1)
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
                    s.addConstr(times[col_j] -
                                times[row_i] >= expected_runtimes[row_i] -
                                (1 - alpha) * M - beta * M)

                    # ti - tj <= ej
                    s.addConstr(times[row_i] -
                                times[col_j] >= expected_runtimes[col_j] -
                                (1 - alpha) * M - (1 - beta) * M)

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == int(pin))

        s.setObjective(MySum(costs), gp.GRB.MAXIMIZE)
        s.optimize()
        end_time = time.time()
        runtime = end_time - start_time
        if log_dir is not None:
            assert log_dir is not None
            raise NotImplementedError

        if s.status == gp.GRB.OPTIMAL:
            self._logger.debug(f"Found optimal value: {s.objVal}")
            assignment = [t.X for t in times], [p.X for p in placements]
            return assignment, s.objVal, runtime
        elif s.status == gp.GRB.INFEASIBLE:
            self._logger.debug("No solution to solver run")
            return None, None, None
        else:
            self._logger.debug(f"Opt solver end with status: {s.status}")
            return None, None, None
