from simulator import *
from workers.worker import Worker
from workers.worker_pool import WorkerPool
from workload.lattice import Lattice
from workload.operator import Operator
from workload.task import Task

op4 = Operator(4, [], 8)
op3 = Operator(3, [4], 6, is_join=True)
op2 = Operator(2, [3], 4)
op1 = Operator(1, [3], 2)
op0 = Operator(0, [1, 2], 10)

lat = Lattice([op0, op1, op2, op3, op4])

print("=" * 50)
print("Test lattice")
print(lat)
# print (lat.get_children(3))

# Lattice {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
# [4]

# ```
# v1 -> v3 -> v4
# ^      ^
# |      |
# v0 ->  v2
# ```

print("=" * 50)
print("Test task queue")

task_queue = []
e = Task(1, 0, 0)
e.start(lat, 0)
for i in range(10):
    e.step()
e.finish(task_queue, lat, 2)
print(e, task_queue)
# (<Task 1; Running Op 0; Available at Time 0; Executed 0 to 3; Deadline: None>,
# [<Task 1; Running Op 1; Available at Time 3; Executed None to None; Deadline: None>,
#  <Task 1; Running Op 2; Available at Time 3; Executed None to None; Deadline: None>])

print("=" * 50)
print("Test worker pool")

w = Worker(1)
w.do_job(Task(7, 4, 2), lat, 2)
print(w)
# GPU Worker 1 --
#   log: [<Task 7; Running Op 4; Available at Time 2; Executed 2 to None; Deadline: None>];
#   curr_task: <Task 7; Running Op 4; Available at Time 2; Executed 2 to None; Deadline: None>

# In[5]:

wp = WorkerPool(1, 0)
print(wp)
# Worker Pool:
# Worker 1: [<Task 7; Running Op 4; Available at Time 2>]

print("=" * 50)
print("Test Fifo with one worker, long period with no overlap")

s = FifoSimulator(1,
                  0, [Task(2, 0, 4), Task(5, 0, 40)],
                  lat,
                  gpu_exact_match=True)
s.simulate(100)
print(s.history())

print("=" * 50)
print("Test Fifo with two worker, long period with no overlap")

s = FifoSimulator(2,
                  0, [Task(2, 0, 4), Task(5, 0, 40)],
                  lat,
                  gpu_exact_match=True)
s.simulate(100)
print(s.history())

print("=" * 50)
print("Test Fifo with two worker, shorter period with overlap")

s = FifoSimulator(2,
                  0, [Task(2, 0, 4), Task(5, 0, 4)],
                  lat,
                  gpu_exact_match=True)
s.simulate(100)
print(s.history())

print("=" * 50)
print("Test Deadlines")
print("=" * 50)
# In[14]:

d_op4 = Operator(4, [], 8, relative_deadline=10)
d_op3 = Operator(3, [4], 6, is_join=True, relative_deadline=6)
d_op2 = Operator(2, [3], 4, relative_deadline=4)
d_op1 = Operator(1, [3], 2, relative_deadline=5)
d_op0 = Operator(0, [1, 2], 10, relative_deadline=10)

d_lat = Lattice([d_op0, d_op1, d_op2, d_op3, d_op4])
print("Test deadline-annotated Lattice")
print(d_lat)

print("=" * 50)
print("Test Fifo with one worker; not schedulable")
task_list = [Task(1, 0, 4, deadline=14)]
s = FifoSimulator(1, 0, task_list, d_lat, gpu_exact_match=True)
s.simulate(100)
print(s.history())

# In[19]:

print("=" * 50)
print("Test EDF with one worker; not schedulable")
task_list = [Task(1, 0, 4, deadline=14)]
s = EdfSimulator(1, 0, task_list, d_lat)
s.simulate(100)
print(s.history())

# In[20]:

print("=" * 50)
print("Test EDF with two worker; schedulable")
task_list = [Task(1, 0, 4, deadline=14)]
s = EdfSimulator(2, 0, task_list, d_lat)
s.simulate(300)
print(s.history())
