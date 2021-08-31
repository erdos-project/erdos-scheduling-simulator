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

    def schedule(self, time: int, task_queue, timeout: int):
        raise NotImplementedError

    def simulate(self, timeout: int, v=0):
        if not self.tasks_list:
            return
        self.tasks_list.sort(key=(lambda e: e.release_time))
        task_queue = []
        for time in range(timeout):
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
            self.schedule(time, task_queue, timeout)
            # finally advance the workers and time
            new_task_queue = []
            for worker in self.worker_pool.workers():
                if worker.current_task:
                    worker.current_task.step()
                    if worker.current_task.time_remaining == 0:
                        worker.current_task.finish(new_task_queue,
                                                   self.lattice, time)
                        print("Finished task: {}".format(worker.current_task))
                        worker.current_task = None
            task_queue.extend(new_task_queue)  # add newly spawned tasks

    def show_running_tasks(self):
        for worker in self.worker_pool.workers():
            if worker.current_task:
                print(f"Worker {worker.unique_id}: {worker.current_task}")

    def history(self):
        return self.worker_pool.history()


class FifoSimulator(Simulator):
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: Lattice,
                 gpu_exact_match: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.gpu_exact_match = gpu_exact_match

    def schedule(self, time: int, task_queue, timeout: int):
        for worker in self.worker_pool.workers():
            if worker.current_task == None and task_queue:
                if self.gpu_exact_match:
                    worker.exact_match_do_job(task_queue, self.lattice, time)
                else:
                    worker.gpu_guarded_do_job(task_queue, self.lattice, time)


class LlfSimulator(Simulator):
    def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
                 lattice: Lattice):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)

    def schedule(self, time: int, task_queue, timeout: int):
        task_queue.sort(key=lambda x: x.deadline - x.time_remaining
                        if x.deadline else timeout)
        for worker in self.worker_pool.workers():
            if worker.current_task == None and task_queue:
                worker.do_job(task_queue.pop(0), self.lattice, time)


class EdfSimulator(Simulator):
    def __init__(self, num_cpus: int, num_gpus: int, tasks_list: List[Task],
                 lattice: Lattice):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)

    def schedule(self, time: int, task_queue, timeout: int):
        task_queue.sort(key=lambda x: x.deadline if x.deadline else timeout)
        for worker in self.worker_pool.workers():
            if worker.current_task == None and task_queue:
                worker.do_job(task_queue.pop(0), self.lattice, time)
