from typing import List

# from z3 import Int, Solver, Implies, Or, IntVal, unsat, Optimize
import gurobipy as gp
from gurobipy import GRB
from schedulers.ilp_scheduler import ILPScheduler
import functools  # for reduce
import time


class GurobiScheduler(ILPScheduler):
<<<<<<< HEAD
    def schedule(
        self,
        needs_gpu: List[bool],
        release_times: List[int],
        absolute_deadlines: List[int],
        expected_runtimes: List[int],
        dependency_matrix,
        pinned_tasks: List[int],
        num_tasks: int,
        num_gpus: int,
        num_cpus: int,
        bits=None,
        optimize=False,
        dump=False,
        dump_nx=False,
        outpath=None,
    ):
        M = max(absolute_deadlines)
=======

    def schedule(self,
                 needs_gpu: List[bool],
                 release_times: List[int],
                 absolute_deadlines: List[int],
                 expected_runtimes: List[int],
                 dependency_matrix,
                 pinned_tasks: List[int],
                 num_tasks: int,
                 num_gpus: int,
                 num_cpus: int,
                 bits=None,
                 optimize=False,
                 dump=False,
                 outpath=None):
        M = 10000000
>>>>>>> b2d566e1cae698824d9814c4e85acdf1245d64a7

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        start_time = time.time()
        # import pdb; pdb.set_trace()
        if optimize:
            print("We are Optimizing")
            s = gp.Model('RAP')
            import IPython
            IPython.embed
            s.setParam("OptimalityTol", 1e-3)
        else:
            print("Missing opt flag")
            import IPython
            IPython.embed()
            raise NotImplementedError
        # x = m.addVar(vtype=GRB.BINARY, name="x")
        times = [
            s.addVar(vtype=GRB.INTEGER, name=f't{i}')
            for i in range(0, num_tasks)
        ]  # Time when execution starts
        costs = [
            s.addVar(vtype=GRB.INTEGER, name=f'c{i}')
            for i in range(0, num_tasks)
        ]  # Costs of gap
        placements = [
            s.addVar(vtype=GRB.INTEGER, name=f'p{i}')
            for i in range(0, num_tasks)
        ]  # placement on CPU or GPU

        for t, r, e, d, c in zip(times, release_times, expected_runtimes,
                                 absolute_deadlines, costs):
            # Finish before deadline.
            s.addConstr(t + e <= d - 1)
            # Start at or after release time.
            s.addConstr(r <= t)
            # cost = s.minimize(d - t - e)
            # cost = s.minimize(d - t - e)
            s.addConstr(d - t - e == c)
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
                disjoint = [
                    times[row_i] + expected_runtimes[row_i] <=
                    times[col_j] - 1,
                    times[col_j] + expected_runtimes[col_j] <= times[row_i] - 1
                ]
                if dependency_matrix[row_i][col_j]:
                    s.addConstr(
                        disjoint[0]
                    )  # dependent jobs need to finish before the next one
                if row_i < col_j:
                    alpha = s.addVar(vtype=GRB.BINARY,
                                     name=f'alpha{row_i}_{col_j}')
                    beta = s.addVar(vtype=GRB.BINARY,
                                    name=f'beta{row_i}_{col_j}')
                    decision_vars.append((alpha, beta))
                    s.addConstr(
                        -alpha * M - beta * M + 1 <=
                        placements[row_i] - placements[col_j])  # pi > pj
                    s.addConstr(
                        alpha * M + (1 - beta) * M - 1 >=
                        placements[row_i] - placements[col_j])  # pj > pi
                    s.addConstr(times[col_j] - times[row_i] >=
                                expected_runtimes[row_i] -
                                (1 - alpha) * M - beta * M)  # tj - ti >= ei
                    s.addConstr(times[row_i] - times[col_j] >=
                                expected_runtimes[col_j] - (1 - alpha) * M -
                                (1 - beta) * M)  # ti - tj <= ej

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == int(pin))

        if optimize:
            s.setObjective(MySum(costs), GRB.MAXIMIZE)
        s.optimize()
        end_time = time.time()
        runtime = end_time - start_time
        if dump:
            assert outpath is not None
            # import IPython; IPython.embed()
            with open(outpath, "w") as outfile:
                outfile.write(s.sexpr())
                if not optimize:
                    outfile.write("(check-sat)")

        if s.status == GRB.OPTIMAL:
            print(f"Found optimal value: {s.objVal}")
            # find model
            assignment = [t.X for t in times], [p.X for p in placements]
            # import IPython; IPython.embed()
            return assignment, s.objVal, runtime
        elif s.status == GRB.INFEASIBLE:
            print("No solution to solver run")
            return None, None, None
        else:

            print(f"Opt solver end with status: {s.status}")
            # import IPython
            # IPython.embed()
            return None, None, None
