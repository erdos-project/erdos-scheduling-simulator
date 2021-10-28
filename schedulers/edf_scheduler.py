from typing import List

from schedulers import BaseScheduler
from workload import Task, TaskGraph


# TODO (Sukrit): This Scheduler does not correspond to the base Scheduler
# interface. Fix this once the interface is defined.
class EDFScheduler(BaseScheduler):
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: TaskGraph,
                 preemptive: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.preemptive = preemptive

    def schedule(self, workload, resources):
        needs_gpu = workload.needs_gpu
        absolute_deadlines = workload.absolute_deadlines
        running_tasks = resources.running_tasks
        num_tasks = resources.num_tasks
        num_gpus = resources.num_gpus
        num_cpus = resources.num_cpus

        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,
                              num_cpus + num_gpus))  # cpus indexed after gpus
        placements = [None] * num_tasks
        assignment = [None] * num_tasks
        priority = [
            absolute_deadlines.index(e) for e in sorted(absolute_deadlines)
        ]

        if not self.preemptive:
            # check and place running tasks first
            placements = running_tasks
            assignment = [
                0 if (t is not None) else None for t in running_tasks
            ]
            gpu_pool = [i for i in gpu_pool if i not in running_tasks]
            cpu_pool = [i for i in cpu_pool if i not in running_tasks]
            # print (assignment, running_tasks)
            # print (gpu_count, cpu_count)
        for index in priority:
            if needs_gpu[index]:
                if len(gpu_pool) > 0 and running_tasks[index] is None:
                    placements[index] = gpu_pool.pop(0)
                    assignment[index] = 0
            else:
                if len(cpu_pool) > 0 and running_tasks[index] is None:
                    placements[index] = cpu_pool.pop(0)
                    assignment[index] = 0
            if len(gpu_pool) + len(cpu_pool) == 0:
                break

        return placements, assignment
