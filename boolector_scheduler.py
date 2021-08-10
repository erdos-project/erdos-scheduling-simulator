from z3 import *
import pyboolector
from pyboolector import Boolector, BoolectorException
import time
def scheduler(requires_gpu, release_times, deadlines, expected_execution, dependency_matrix, pinned, num_tasks: int, GPUS: int, CPUS: int, bits : int = 8):

  s = Boolector()
  s.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, True)

  bvsort = s.BitVecSort(bits)

  times = [s.Var(bvsort, f't{i}') for i in range(0,num_tasks)] #Time when execution starts
  placements = [s.Var(bvsort, f'p{i}') for i in range (0,num_tasks)] # placement on CPU or GPU
  release_times = [s.Const(num, bits) for num in release_times]
  deadlines = [s.Const(num, bits) for num in deadlines]
  expected_execution = [s.Const(num,bits) for num in expected_execution]



  for t, r, e, d  in zip(times, release_times, expected_execution, deadlines):
    s.Assert(t < d - e) # finish before deadline
    s.Assert( t > r) # start at or after release

  CPUS = s.Const(CPUS, bits)
  GPUS = s.Const(GPUS, bits)
  for p, gpu in zip(placements, requires_gpu):
    if gpu:
      s.Assert( p > CPUS)
      s.Assert( p <= (GPUS  + CPUS)) # second half is GPUS
    else:
      s.Assert(p > s.Const(0,bits))
      s.Assert ( p <= CPUS) # first half is CPUS

  for row_i in range(len(dependency_matrix)):
    for col_j in range(len(dependency_matrix[0])):
      disjoint = [(times[row_i] + expected_execution[row_i]) < times[col_j], times[col_j] + expected_execution[col_j]<  times[row_i] ]
      if dependency_matrix[row_i][col_j]:
        s.Assert(disjoint[0]) # dependent jobs need to finish before the next one
      if row_i != col_j:
        s.Assert(s.Implies(placements[row_i] == placements[col_j], s.Or(disjoint[0], disjoint[1]) )) # cannot overlap if on the same hardware
  for i, pin in enumerate(pinned):
    if pin:
      s.Assert(placement[i] ==  s.Constant(pin, bits))
  #s.Simplify()
  #s.Dump('smt2', 'tmp.smt')


  schedulable = s.Sat()
  if schedulable == s.SAT:
    ass_p, ass_t = [int(p.assignment,2) for p in placements], [int(t.assignment,2) for t in times]
    print (ass_p, ass_t)
    l = list(zip(ass_p, ass_t))
    l.sort()
    return l
  return None



HORIZON= 45
TASKS = 5
GPUS = 2
CPUS = 10

requires_gpu = [True ] * TASKS # True if requires GPU
release_times = [2 for _ in range (0,TASKS)] # release time when task is allowed to start
deadlines = [HORIZON for _ in range(0,TASKS)] # deadline when task must finish
expected_execution = [20 for _ in range(0,TASKS)] # expected runtime
dependency_matrix = [[ False for i in range (0,TASKS)] for j in range( 0,TASKS)] # True if task i must finish before task j starts
pinned = [None] * TASKS # hardware index if task is pinned to that machine (or already running there)


dependency_matrix[0][1] = True
requires_gpu[3] = False

#start = time.time()
#out = scheduler(requires_gpu, release_times, deadlines, expected_execution, dependency_matrix, pinned, TASKS, GPUS, CPUS)
#end = time.time()
#print(end - start)


#print (out)




def run(MULTIPLIER, expected_runtime = 5):
  HORIZON= 50 * MULTIPLIER
  TASKS = 5*MULTIPLIER
  GPUS = 2
  CPUS = 10

  print (f"Running for {TASKS} task over horizon of {HORIZON}")


  requires_gpu = [True ] * TASKS # True if requires GPU
  release_times = [2 for _ in range (0,TASKS)] # release time when task is allowed to start
  deadlines = [HORIZON for _ in range(0,TASKS)] # deadline when task must finish
  expected_execution = [expected_runtime for _ in range(0,TASKS)] # expected runtime :: 20 is best schedulable
  dependency_matrix = [[ False for i in range (0,TASKS)] for j in range( 0,TASKS)] # True if task i must finish before task j starts
  pinned = [None] * TASKS # hardware index if task is pinned to that machine (or already running there)


  dependency_matrix[0][1] = True
  requires_gpu[3] = False

  start = time.time()
  out = scheduler(requires_gpu, release_times, deadlines, expected_execution, dependency_matrix, pinned, TASKS, GPUS, CPUS, bits = 10)
  end = time.time()
  print(end - start)
#  print ('GPU/CPU Placement:' ,out[0]) 
#  print ('Start Time:' ,out[1])
  print (out)
  return (end-start)
#run_time_5 = [(5*i, run(i, 5)) for i in range(1,21, 1)]
#run_time_10 = [(5*i, run(i, 10)) for i in range(1,21, 1)]
#print (run_time_5)
#print (run_time_10)
if __name__ == "__main__": 
  run_time_15 = [(5*i, run(i, 15)) for i in range(1,11, 1)]
  print (run_time_15)



