import time
from typing import List

import pyboolector
from pyboolector import Boolector, BoolectorException

from schedulers.ilp_scheduler import ILPScheduler


class BoolectorScheduler(ILPScheduler):
    def scheduler(self,
                  needs_gpu: List[bool],
                  release_times: List[int],
                  absolute_deadlines: List[int],
                  expected_runtimes: List[int],
                  dependency_matrix,
                  pinned_tasks: List[int],
                  num_tasks: int,
                  num_gpus: int,
                  num_cpus: int,
                  bits: int = 8):
        s = Boolector()

        s.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, True)

        bvsort = s.BitVecSort(bits)

        # Time when task execution starts.
        start_times = [s.Var(bvsort, f't{i}') for i in range(0, num_tasks)]
        # Task placement on CPU or GPU.
        placements = [s.Var(bvsort, f'p{i}') for i in range(0, num_tasks)]
        release_times = [s.Const(num, bits) for num in release_times]
        absolute_deadlines = [s.Const(num, bits) for num in absolute_deadlines]
        expected_runtimes = [s.Const(num, bits) for num in expected_runtimes]

        for t, r, e, d in zip(start_times, release_times, expected_runtimes,
                              absolute_deadlines):
            # Finish before deadline.
            s.Assert(t < d - e)
            # Start at or after release time.
            s.Assert(t > r)

        num_cpus = s.Const(num_cpus, bits)
        num_gpus = s.Const(num_gpus, bits)
        for p, gpu in zip(placements, needs_gpu):
            if gpu:
                s.Assert(p > num_cpus)
                s.Assert(p <= (num_gpus + num_cpus))  # second half is num_gpus
            else:
                s.Assert(p > s.Const(0, bits))
                s.Assert(p <= num_cpus)  # first half is num_cpus

        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [(start_times[row_i] + expected_runtimes[row_i]) <
                            start_times[col_j],
                            start_times[col_j] + expected_runtimes[col_j] <
                            start_times[row_i]]
                if dependency_matrix[row_i][col_j]:
                    # Dependent tasks need to finish before the next one.
                    s.Assert(disjoint[0])
                if row_i != col_j:
                    # Cannot overlap if on the same hardware.
                    s.Assert(
                        s.Implies(placements[row_i] == placements[col_j],
                                  s.Or(disjoint[0], disjoint[1])))

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.Assert(placement[i] == s.Constant(pin, bits))
        #s.Simplify()
        #s.Dump('smt2', 'tmp.smt')

        schedulable = s.Sat()
        if schedulable == s.SAT:
            ass_p, ass_t = [int(p.assignment, 2) for p in placements
                            ], [int(t.assignment, 2) for t in start_times]
            print(ass_p, ass_t)
            l = list(zip(ass_p, ass_t))
            l.sort()
            return l
        return None
