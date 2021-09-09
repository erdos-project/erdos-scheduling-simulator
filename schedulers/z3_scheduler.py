from typing import List

from z3 import Int, Solver, Implies, Or, IntVal, unsat

from schedulers.ilp_scheduler import ILPScheduler


class Z3Scheduler(ILPScheduler):
    def schedule(self, needs_gpu: List[bool], release_times: List[int],
                 absolute_deadlines: List[int], expected_runtimes: List[int],
                 dependency_matrix, pinned_tasks: List[int], num_tasks: int,
                 num_gpus: int, num_cpus: int):
        times = [Int(f't{i}')
                 for i in range(0, num_tasks)]  # Time when execution starts
        placements = [Int(f'p{i}')
                      for i in range(0, num_tasks)]  # placement on CPU or GPU
        s = Solver()
        for t, r, e, d in zip(times, release_times, expected_runtimes,
                              absolute_deadlines):
            # Finish before deadline.
            s.add(t + e < d)
            # Start at or after release time.
            s.add(r <= t)
        for p, gpu in zip(placements, needs_gpu):
            if gpu:
                s.add(p > num_cpus, p <=
                      (num_gpus + num_cpus))  # second half is num_gpus
            else:
                s.add(p > 0, p <= num_cpus)  # first half is num_cpus

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
                if row_i != col_j:
                    s.add(
                        Implies(placements[row_i] == placements[col_j],
                                Or(disjoint))
                    )  # cannot overlap if on the same hardware

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == IntVal(pin))

        schedulable = s.check()
        print(schedulable)
        if schedulable != unsat:
            return [s.model()[p]
                    for p in placements], [s.model()[t] for t in times]
        return None
