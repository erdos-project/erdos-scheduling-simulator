from typing import List

import functools
from z3 import Int, Solver, Implies, Or, IntVal, unsat, Optimize

from schedulers.ilp_scheduler import ILPScheduler

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Z3Scheduler(ILPScheduler):
    def schedule(self, needs_gpu: List[bool], release_times: List[int],
                 absolute_deadlines: List[int], expected_runtimes: List[int],
                 dependency_matrix, pinned_tasks: List[int], num_tasks: int,
                 num_gpus: int, num_cpus: int, bits = None, optimize=False,
                 dump=False, outpath = None,
                 dump_nx=False, nx_outpath=None):
        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        # Add new constraint node 'new_node' with attribute 'relation'
        # e.g. call add_relation(G, 'ub_a', '<', ['a', -5]) to add constraint (a < 5)
        def add_relation(G, new_node, relation, node_lst, weight_lst=None):
            if weight_lst is None: 
                weight_lst = [1 if relation in ['<', '<=', '>', '>=', '='] else None] * len(node_lst)
            G.add_node(new_node, typ=relation)
            for n, w in zip(node_lst, weight_lst):
                G.add_node(n)
                G.add_edge(n, new_node, w=w)

        times = [Int(f't{i}')
                 for i in range(0, num_tasks)]  # Time when execution starts
        costs = [Int(f'c{i}')for i in range(0, num_tasks)]  # Costs of gap
        placements = [Int(f'p{i}')
                      for i in range(0, num_tasks)]  # placement on CPU or GPU

        if dump_nx:
            G = nx.Graph()
            G.add_nodes_from([f't{i}' for i in range(num_tasks)])
            G.add_nodes_from([f'p{i}' for i in range(num_tasks)])
            
        if optimize:
            print ("We are Optimizing")
            s = Optimize()
        else:
            s = Solver() 
        for i, (t, r, e, d, c) in enumerate(zip(times, release_times, expected_runtimes,
                              absolute_deadlines, costs)):
            # Finish before deadline.
            s.add(t + e < d)
            # Start at or after release time.
            s.add(r <= t)

            if dump_nx:
                # Finish before deadline.
                add_relation(G, f'tu{i}', '<', [f't{i}', e-d])
                # Start at or after release time.
                add_relation(G, f'tl{i}', '>=', [f't{i}', -r])

        for i, (p, gpu) in enumerate(zip(placements, needs_gpu)):
            if gpu:
                s.add(p > num_cpus, p <=
                      (num_gpus + num_cpus))  # second half is num_gpus
            else:
                s.add(p > 0, p <= num_cpus)  # first half is num_cpus

            if dump_nx:
                ub = num_gpus + num_cpus if gpu else num_cpus
                lb = num_cpus if gpu else 0
                add_relation(G, f'pu{i}', '<=', [f'p{i}', -ub])
                add_relation(G, f'pl{i}', '>', [f'p{i}', -lb])


        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [
                    times[row_i] + expected_runtimes[row_i] < times[col_j],
                    times[col_j] + expected_runtimes[col_j] < times[row_i]
                ]
                if dependency_matrix[row_i][col_j]:
                    s.add(
                        disjoint[0]
                    )  # dependent jobs need to finish before the next one
                    if dump_nx:
                        add_relation(G, f'dep_{row_i}_{col_j}', '<', 
                                [f't{row_i}', f't{col_j}', expected_runtimes[row_i]], [1,-1,1])
                if row_i != col_j: # and row_i < col_j ?
                    s.add(
                        Implies(placements[row_i] == placements[col_j],
                                Or(disjoint))
                    )  # cannot overlap if on the same hardware
                    if dump_nx:
                        neq    = f'eq_{row_i}_{col_j}'
                        dep_ij = f'dep_{row_i}_{col_j}'
                        dep_ji = f'dep_{col_j}_{row_i}'
                        or_dep = f'or_dep_{row_i}_{col_j}'
                        disjoint = f'dj_{row_i}_{col_j}'
                        add_relation(G, neq   , '!=', [f'p{row_i}', f'p{col_j}'])
                        add_relation(G, dep_ij, '<' , [f't{row_i}', f't{col_j}', expected_runtimes[row_i]], [1,-1,1])
                        add_relation(G, dep_ji, '<' , [f't{col_j}', f't{row_i}', expected_runtimes[col_j]], [1,-1,1])
                        add_relation(G, or_dep, 'or', [dep_ij, dep_ji])
                        add_relation(G, disjoint, 'or', [neq, or_dep])

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == IntVal(pin))
                if dump_nx:
                    add_relation(G, f'pin_{i}', '==', [f'p{i}', pin])

        if optimize:
            result = s.maximize(MySum(costs))
        if dump:
            assert outpath is not None
            # import IPython; IPython.embed()
            with open (outpath,"w") as outfile:
                outfile.write(s.sexpr())
                if not optimize: 
                    outfile.write("(check-sat)")
        if dump_nx:
            nx.write_gpickle(G, outpath+'.pkl')
            plt.savefig(outpath+".png")
            # import pdb; pdb.set_trace()

        schedulable = s.check()
        if optimize: 
            print(s.lower(result))
        print(schedulable)
        if schedulable != unsat:
            return [s.model()[p]
                    for p in placements], [s.model()[t] for t in times]
        return None
