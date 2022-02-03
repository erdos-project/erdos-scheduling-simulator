from typing import List

import functools
from z3 import Int, Solver, Implies, If, Or, IntVal, unsat, Optimize, BitVec,Bool, BitVecVal, UGE, ULT
from math import ceil, log2

from schedulers.ilp_scheduler import ILPScheduler
import time

from typing import Dict


class Z3Scheduler(ILPScheduler):

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
                 optimize=False,
                 log_dir=None,
                 verbose=True):
        """
        Runs scheduling using Z3
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
            num_resources (`Dict[str, int]`): Number of resources.
            bits (`int`): Number of bits to use for the ILP.
            optimize (`bool`): Run with optimization or just feasible.
            log_dir (`str`): Directory to write the ILP to.
            verbose (`bool`): print status update.
        """
        
        num_gpus = num_resources['GPU']
        num_cpus = num_resources['CPU']

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        # Add new constraint node 'new_node' with attribute 'attr'
        # e.g. call add_relation(G, 'ub_a', {'rel:'<', 'num':5}, ['a'])
        #  to add constraint (a < 5)
        
        bits = ceil(log2(num_gpus + num_cpus))

        start_time = time.time()
        times = [Int(f't{i}')
                 for i in range(0, num_tasks)]  # Time when execution starts
        costs = [Int(f'c{i}') for i in range(0, num_tasks)]  # Costs of gap
        placements = [Int(f'p{i}')
                      for i in range(0, num_tasks)]  # placement on CPU or GPU

        if optimize:
            if verbose:
                print("We are Optimizing")
            s = Optimize()
        else:
            s = Solver()

        
        # Add constraints
        for i, (t, r, e, d, c) in enumerate(
                zip(times, release_times, expected_runtimes,
                    absolute_deadlines, costs)):
            # Finish before deadline.
            s.add(t + e <= d)
            # Start at or after release time.
            s.add(r <= t)
            s.add(c == d - e - t)

        for i, (p, gpu) in enumerate(zip(placements, needs_gpu)):
            if gpu:
                placements[i] = Bool(f'p{i}')
                # gpu_bits = ceil(log2(num_gpus))
                # placements[i] = BitVecVal(num_cpus, gpu_bits) 
                # s.add(UGE(p , BitVecVal(num_cpus, gpu_bits)), ULT(p ,
                #       BitVecVal(num_gpus + num_cpus, gpu_bits)))  # second half is num_gpus
            else:
                s.add(p > 0, p <= num_cpus)
                # cpu_bits = ceil(log2(num_cpus))
                # placements[i] = BitVecVal(num_cpus, cpu_bits)
                # s.add(UGE(p, BitVecVal(0, cpu_bits)), ULT (p , BitVecVal(num_cpus, cpu_bits)))  # first half is num_cpus

        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [
                    times[row_i] + expected_runtimes[row_i] <= times[col_j],
                    times[col_j] + expected_runtimes[col_j] <= times[row_i]
                ]
                if dependency_matrix[row_i][col_j]:
                    s.add(
                        disjoint[0]
                    )  # dependent jobs need to finish before the next one

                if row_i != col_j:  # and row_i < col_j ?
                    if needs_gpu[row_i]:
                        i_val = If(placements[row_i],IntVal(10),IntVal(11))
                    else:
                        i_val = placements[row_i]
                    if needs_gpu[col_j]:
                        j_val = If(placements[col_j],IntVal(10),IntVal(11))
                    else:
                        j_val = placements[col_j]
                    s.add(
                        # ITE(placements[row_i],10,11)) == ITE(placements[col_j], 10,11))
                        # Implies(placements[row_i] == placements[col_j],
                        #         Or(disjoint))
                        Implies(i_val == j_val, Or(disjoint))
                    )  # cannot overlap if on the same hardware

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == BitVecVal(pin , bits))

        if optimize:
            result = s.maximize(MySum(costs))

        

        if log_dir is not None:
            log_dir = log_dir + f"{'opt' if optimize else 'feas'}.smt"
            with open(log_dir, "w") as outfile:
                outfile.write(s.sexpr())
                if not optimize:
                    outfile.write("(check-sat)")
        
        schedulable = s.check()
        end_time = time.time()
        runtime = end_time - start_time
        cost = None
        if optimize:
            cost = s.lower(result)
            if verbose:
                print(cost)
        if verbose:
            print(schedulable)
        if schedulable != unsat:
            placements= [int(str(s.model()[p])) if not needs_gpu[i] else (10 if bool(s.model()[p]) else 11)for i,p in enumerate(placements) ]
            outputs = [int(str(s.model()[t])) for t in times
                       ], [p for p in placements]
            if verbose:
                print(outputs)
            return outputs, cost, runtime
        return (None, None), None, runtime



    def __schedule(self,
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
                 log_dir=None,
                 verbose=True):
        """
        Runs scheduling using Z3
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
            num_gpus (`int`): Number of GPUs.
            num_cpus (`int`): Number of CPUs.
            bits (`int`): Number of bits to use for the ILP.
            optimize (`bool`): Run with optimization or just feasible.
            log_dir (`str`): Directory to write the ILP to.
            verbose (`bool`): print status update.
        """

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        # Add new constraint node 'new_node' with attribute 'attr'
        # e.g. call add_relation(G, 'ub_a', {'rel:'<', 'num':5}, ['a'])
        #  to add constraint (a < 5)

        start_time = time.time()
        times = [Int(f't{i}')
                 for i in range(0, num_tasks)]  # Time when execution starts
        costs = [Int(f'c{i}') for i in range(0, num_tasks)]  # Costs of gap
        placements = [Int(f'p{i}')
                      for i in range(0, num_tasks)]  # placement on CPU or GPU

        if optimize:
            if verbose:
                print("We are Optimizing")
            s = Optimize()
        else:
            s = Solver()
        for i, (t, r, e, d, c) in enumerate(
                zip(times, release_times, expected_runtimes,
                    absolute_deadlines, costs)):
            # Finish before deadline.
            s.add(t + e <= d)
            # Start at or after release time.
            s.add(r <= t)
            s.add(c == d - e - t)

        for i, (p, gpu) in enumerate(zip(placements, needs_gpu)):
            if gpu:
                s.add(p > num_cpus, p <=
                      (num_gpus + num_cpus))  # second half is num_gpus
            else:
                s.add(p > 0, p <= num_cpus)  # first half is num_cpus

        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [
                    times[row_i] + expected_runtimes[row_i] <= times[col_j],
                    times[col_j] + expected_runtimes[col_j] <= times[row_i]
                ]
                if dependency_matrix[row_i][col_j]:
                    s.add(
                        disjoint[0]
                    )  # dependent jobs need to finish before the next one

                if row_i != col_j:  # and row_i < col_j ?
                    s.add(
                        Implies(placements[row_i] == placements[col_j],
                                Or(disjoint))
                    )  # cannot overlap if on the same hardware

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == IntVal(pin))

        if optimize:
            result = s.maximize(MySum(costs))

        end_time = time.time()
        runtime = end_time - start_time

        if log_dir is not None:
            log_dir = log_dir + f"{'opt' if optimize else 'feas'}.smt"
            with open(log_dir, "w") as outfile:
                outfile.write(s.sexpr())
                if not optimize:
                    outfile.write("(check-sat)")

        schedulable = s.check()
        cost = None
        if optimize:
            cost = s.lower(result)
            if verbose:
                print(cost)
        if verbose:
            print(schedulable)
        if schedulable != unsat:
            outputs = [int(str(s.model()[t])) for t in times
                       ], [int(str(s.model()[p])) for p in placements]
            if verbose:
                print(outputs)
            return outputs, cost, runtime
        return (None, None), None, runtime
