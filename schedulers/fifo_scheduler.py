from typing import List

from schedulers import BaseScheduler
from workload import Task, TaskGraph


# TODO (Sukrit): This Scheduler does not correspond to the base Scheduler
# interface. Fix this once the interface is defined.
class FIFOScheduler(BaseScheduler):
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: TaskGraph,
                 gpu_exact_match: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.gpu_exact_match = gpu_exact_match

    def schedule(self, workload, resources):
        num_gpus = resources.num_gpus
        num_cpus = resources.num_cpus
        running_tasks = resources.running_tasks
        needs_gpu = workload.needs_gpu

        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,
                              num_cpus + num_gpus))  # cpus indexed after gpus
        placements = running_tasks
        assignment = [0 if (t is not None) else None for t in running_tasks]
        gpu_pool = [i for i in gpu_pool if i not in running_tasks]
        cpu_pool = [i for i in cpu_pool if i not in running_tasks]

        for i, gpu in enumerate(needs_gpu):
            if gpu:
                if len(gpu_pool) > 0 and running_tasks[i] is None:
                    w_idx = gpu_pool.pop(0)
                    placements[i] = w_idx
                    assignment[i] = 0
            else:
                if len(cpu_pool) > 0 and running_tasks[i] is None:
                    w_idx = cpu_pool.pop(0)
                    placements[i] = w_idx
                    assignment[i] = 0
            if len(gpu_pool) + len(cpu_pool) == 0:
                break

        return placements, assignment
