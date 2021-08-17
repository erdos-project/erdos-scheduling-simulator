from z3 import *
import time


def scheduler(requires_gpu: bool, release_times, deadlines, expected_execution,
              dependency_matrix, pinned, num_tasks: int, GPUS: int, CPUS: int):
    times = [Int(f't{i}')
             for i in range(0, num_tasks)]  #Time when execution starts
    placements = [Int(f'p{i}')
                  for i in range(0, num_tasks)]  # placement on CPU or GPU
    s = Solver()
    for t, r, e, d in zip(times, release_times, expected_execution, deadlines):
        s.add(t + e < d)  # finish before deadline
        s.add(r <= t)  # start at or after release
    for p, gpu in zip(placements, requires_gpu):
        if gpu:
            s.add(p > CPUS, p <= (GPUS + CPUS))  # second half is GPUS
        else:
            s.add(p > 0, p <= CPUS)  # first half is CPUS

    for row_i in range(len(dependency_matrix)):
        for col_j in range(len(dependency_matrix[0])):
            disjoint = [
                times[row_i] + expected_execution[row_i] < times[col_j],
                times[col_j] + expected_execution[col_j] < times[row_i]
            ]
            if dependency_matrix[row_i][col_j]:
                s.add(disjoint[0]
                      )  # dependent jobs need to finish before the next one
            if row_i != col_j:
                s.add(
                    Implies(placements[row_i] == placements[col_j], Or(
                        disjoint)))  # cannot overlap if on the same hardware
    for i, pin in enumerate(pinned):
        if pin:
            s.add(placement[i] == IntVal(pin))

    schedulable = s.check()
    print(schedulable)
    if schedulable != unsat:
        return [s.model()[p]
                for p in placements], [s.model()[t] for t in times]
    return None


HORIZON = 45
TASKS = 5
GPUS = 2
CPUS = 10

requires_gpu = [True] * TASKS  # True if requires GPU
release_times = [2 for _ in range(0, TASKS)
                 ]  # release time when task is allowed to start
deadlines = [HORIZON
             for _ in range(0, TASKS)]  # deadline when task must finish
expected_execution = [20 for _ in range(0, TASKS)]  # expected runtime
dependency_matrix = [[False for i in range(0, TASKS)] for j in range(0, TASKS)
                     ]  # True if task i must finish before task j starts
pinned = [
    None
] * TASKS  # hardware index if task is pinned to that machine (or already running there)

dependency_matrix[0][1] = True
requires_gpu[3] = False

start = time.time()
out = scheduler(requires_gpu, release_times, deadlines, expected_execution,
                dependency_matrix, pinned, TASKS, GPUS, CPUS)
end = time.time()
print(end - start)

print(out)


def run(MULTIPLIER):
    HORIZON = 50 * MULTIPLIER
    TASKS = 5 * MULTIPLIER
    GPUS = 2
    CPUS = 10

    requires_gpu = [True] * TASKS  # True if requires GPU
    release_times = [2 for _ in range(0, TASKS)
                     ]  # release time when task is allowed to start
    deadlines = [HORIZON
                 for _ in range(0, TASKS)]  # deadline when task must finish
    expected_execution = [20 for _ in range(0, TASKS)]  # expected runtime
    dependency_matrix = [[False for i in range(0, TASKS)]
                         for j in range(0, TASKS)
                         ]  # True if task i must finish before task j starts
    pinned = [
        None
    ] * TASKS  # hardware index if task is pinned to that machine (or already running there)

    dependency_matrix[0][1] = True
    requires_gpu[3] = False

    start = time.time()
    out = scheduler(requires_gpu, release_times, deadlines, expected_execution,
                    dependency_matrix, pinned, TASKS, GPUS, CPUS)
    end = time.time()
    print(end - start)
    print(out)
    return (end - start)


[run(i) for i in range(1, 7, 1)]
