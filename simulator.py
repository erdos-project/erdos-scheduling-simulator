from copy import deepcopy
from typing import List

from workers.worker_pool import WorkerPool
from workload.lattice import Lattice
from workload.task import Task


class Emulator:
    def __init__(self):
        self.assignment = []  # list of worker_id X task_id
        self.release = 0  # time when assignment is made available

    def set_schedule(schedule,
                     task_set: List[Task],
                     worker_pool: WorkerPool,
                     dependencies,
                     timeout: int,
                     v=0):
        """
        Args:
            schedule: Scheduling function.
            task_set: List of tasks to schedule.
            worker_pool: Workers to be given tasks.
            dependencies: Matrix of dependencies between tasks
        """
        start = time.time()
        self.assignments, self.release = schedule(task_set, worker_pool,
                                                  dependencies, timeout)
        end = time.time()

    def get_schedule(curr_time: int):
        if curr_time >= self.release:
            return self.assignment
        else:
            return None

    def reset():
        self.__init__()


class Simulator:
    def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
                 lattice: Lattice):
        """
        Args:
            tasks_list: List of Tasks.
            lattice: Default one would just have operators not connected to each other
        """
        self.tasks_list = tasks_list.copy()
        self.worker_pool = WorkerPool(num_cpus, num_gpus)
        self.lattice = deepcopy(lattice)
        self.emulator = Emulator()
        self.current_time = 0
        self.current_tasks = []

    def schedule(self, needs_gpu: List[bool], 
            release_times: List[int], 
            absolute_deadlines: List[int], 
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int], 
            running_tasks: List[int], # indicates the worker it's running on None otherwise
            num_tasks: int, 
            num_gpus: int,
            num_cpus: int
            ):
        raise NotImplementedError

    # def schedule(self, time: int, task_queue, timeout: int):
    #     raise NotImplementedError

    def simulate(self, timeout: int, v=0):
        if not self.tasks_list:
            return
        self.tasks_list.sort(key=(lambda e: e.release_time))
        task_queue = []
        time = 0
        while time < timeout :
            if v > 0 and time % 100 == 0:
                print("step: {}".format(time))
            if v > 1:
                print("step: {}".format(time))
            # first determine if there's new tasks to be made visible
            while len(self.tasks_list
                      ) != 0 and self.tasks_list[0].release_time == time:
                task = self.tasks_list.pop(0)
                print("Activate: {}".format(task))
                task_queue.append(task)
            # second determine where to place tasks
            # self.schedule(time, task_queue, timeout)
            running_tasks = self.worker_pool.get_running_tasks()
            runnable = running_tasks + [(None,t)for t in task_queue]
            # print (runnable)
            placement, assignment = self.schedule( 
                [ t.needs_gpu for _, t in runnable ], 
                [ t.release_time - time if t.release_time != None else None for _,t in runnable ], 
                [ t.deadline - time if t.deadline != None else None for _, t in runnable ], 
                [ t.time_remaining for _, t in runnable ],
                None, # dependencies
                [ None for _,t in runnable ], #pinned 
                [ worker for worker,_ in runnable ], # indicates the worker it's running on None otherwise
                len(runnable), 
                self.worker_pool.num_gpus,
                self.worker_pool.num_cpus
            )

            
            # place jobs task on worker
            task_queue = []
            for i,(p,a) in enumerate(zip (placement, assignment)):
                _,t = runnable[i]
                if a == 0: 
                    w = self.worker_pool.get_worker(p)
                    if w.current_task == None or w.current_task.unique_id != t.unique_id:
                        w.do_job(t,self.lattice, time)
                else: 
                    task_queue.append(t)

            # determine step_size as next release event or time remaining
            step_size, step_size_rel = None, None
            if len(self.worker_pool.get_running_tasks()) > 0:
                step_size = min ([t.time_remaining for _, t in self.worker_pool.get_running_tasks()])
            
            if self.worker_pool.has_free_worker() and len(self.tasks_list)>0:
                step_size_rel =  self.tasks_list[0].release_time - time 
                if step_size == None:
                    step_size = step_size_rel 
                else:
                    step_size = min (step_size, step_size_rel)   
            if step_size_rel == None and step_size == None:
                step_size = 1      
            # import pdb; pdb.set_trace()

            # finally advance the workers and time
            new_task_queue = []
            for worker in self.worker_pool.workers():
                if worker.current_task:
                    worker.current_task.step(step_size=step_size)
                    if worker.current_task.time_remaining == 0:
                        worker.current_task.finish(new_task_queue,
                                                   self.lattice, time+step_size)
                        print("Finished task: {}".format(worker.current_task))
                        worker.current_task = None
            task_queue.extend(new_task_queue)  # add newly spawned tasks
            time += step_size

    def show_running_tasks(self):
        for worker in self.worker_pool.workers():
            if worker.current_task:
                print(f"Worker {worker.unique_id}: {worker.current_task}")

    def history(self):
        return self.worker_pool.history()


# class FifoSimulator(Simulator):
#     def __init__(self,
#                  num_cpus: int,
#                  num_gpus: int,
#                  tasks_list: List[Task],
#                  lattice: Lattice,
#                  gpu_exact_match: bool = False):
#         super().__init__(num_cpus, num_gpus, tasks_list, lattice)
#         self.gpu_exact_match = gpu_exact_match
    
#     def schedule(self, time: int, task_queue, timeout: int):
#         for worker in self.worker_pool.workers():
#             if worker.current_task == None and task_queue:
#                 if self.gpu_exact_match:
#                     worker.exact_match_do_job(task_queue, self.lattice, time)
#                 else:
#                     worker.gpu_guarded_do_job(task_queue, self.lattice, time)


# class LlfSimulator(Simulator):
#     def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
#                  lattice: Lattice):
#         super().__init__(num_cpus, num_gpus, tasks_list, lattice)

#     def schedule(self, time: int, task_queue, timeout: int):
#         task_queue.sort(key=lambda x: x.deadline - x.time_remaining
#                         if x.deadline else timeout)
#         for worker in self.worker_pool.workers():
#             if worker.current_task == None and task_queue:
#                 worker.do_job(task_queue.pop(0), self.lattice, time)


# class EdfSimulator(Simulator):
#     def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
#                  lattice: Lattice):
#         super().__init__(num_cpus, num_gpus, tasks_list, lattice)

#     def schedule(self, time: int, task_queue, timeout: int):
#         task_queue.sort(key=lambda x: x.deadline if x.deadline else timeout)
#         for worker in self.worker_pool.workers():
#             if worker.current_task == None and task_queue:
#                 worker.do_job(task_queue.pop(0), self.lattice, time)


class EdfSimulator(Simulator):   
    def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
                 lattice: Lattice, preemptive:bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.preemptive = preemptive
    def schedule(self, needs_gpu: List[bool], 
            release_times: List[int], 
            absolute_deadlines: List[int], 
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int], 
            running_tasks: List[int], # indicates the worker it's running on None otherwise
            num_tasks: int, 
            num_gpus: int,
            num_cpus: int
            ):

        
        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,num_cpus+num_gpus)) #cpus indexed after gpus
        placements = [None]*num_tasks
        assignment = [None]*num_tasks
        priority = [absolute_deadlines.index(e) for e in sorted(absolute_deadlines)]
        
        if not self.preemptive: 
            # check and place running tasks first
            placements = running_tasks
            assignment = [ 0 if (t != None) else None for t in running_tasks]
            gpu_pool = [i for i in gpu_pool if i not in running_tasks]
            cpu_pool = [i for i in cpu_pool if i not in running_tasks]
            # print (assignment, running_tasks)
            # print (gpu_count, cpu_count)
        for index in priority: 
            if needs_gpu[index]:
                if len(gpu_pool) > 0 and running_tasks[index]==None:
                    placements[index] = gpu_pool.pop(0)
                    assignment[index] =  0
            else:
                if len(cpu_pool) > 0 and running_tasks[index]==None:
                    placements[index] = cpu_pool.pop(0)
                    assignment[index] = 0
            if len(gpu_pool) + len(cpu_pool) == 0: 
                break
        
        return placements, assignment


class FifoSimulator(Simulator):
    def __init__(self,
                num_cpus: int,
                num_gpus: int,
                tasks_list: List[Task],
                lattice: Lattice,
                gpu_exact_match: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.gpu_exact_match = gpu_exact_match

    def schedule(self, needs_gpu: List[bool], 
            release_times: List[int], 
            absolute_deadlines: List[int], 
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int], 
            running_tasks: List[int], # indicates the worker it's running on None otherwise
            num_tasks: int, 
            num_gpus: int,
            num_cpus: int
            ):

        
        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,num_cpus+num_gpus)) #cpus indexed after gpus
        placements = running_tasks
        assignment = [ 0 if (t != None) else None for t in running_tasks]
        gpu_pool = [i for i in gpu_pool if i not in running_tasks]
        cpu_pool = [i for i in cpu_pool if i not in running_tasks]
        
        for i,gpu in enumerate (needs_gpu):
            if gpu:
                if len(gpu_pool) > 0 and running_tasks[i] == None:
                    w_idx = gpu_pool.pop(0)
                    placements[i]= w_idx
                    assignment[i] = 0
            else:
                if len(cpu_pool) > 0 and running_tasks[i] == None:
                    w_idx = cpu_pool.pop(0)
                    placements[i]= w_idx
                    assignment[i] = 0
            if len(gpu_pool) + len(cpu_pool) ==0:
                break
        
        return placements, assignment