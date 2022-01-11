from typing import List
from schedulers.base_scheduler import BaseScheduler
from typing import Sequence, Mapping, Tuple, Optional

class ILPScheduler(object):
    def schedule(needs_gpu: List[bool],
                 release_times: List[int],
                 absolute_deadlines: List[int],
                 expected_runtimes: List[int],
                 dependency_matrix,
                 pinned_tasks: List[int],
                 num_tasks: int,
                 num_gpus: int,
                 num_cpus: int,
                 bits: int = 8):
        raise NotImplementedError


class ILPBaseScheduler(BaseScheduler):
    def __init__(self, sched_solver: ILPScheduler):
        self.sched_solver = sched_solver()

    def schedule(self, tasks, task_graph, worker_pools)-> Tuple[float,Sequence[Tuple[int,int]]]:
        dependency_matrix = task_graph.get_dep(tasks)
        needs_gpu = [t.needs_gpu for t in tasks]
        absolute_deadlines = [t.deadline for t in tasks]
        expected_runtimes = [t.expected_runtime for t in tasks]
        num_tasks = len(tasks)

        num_gpus = worker_pools.num_gpu()
        num_cpus = worker_pools.num_cpu()

        release_times = [0] * num_tasks
        pinned_tasks = [None] * num_tasks

        (start_times,
         placements), opt_value, sched_runtime = self.sched_solver.schedule(
             needs_gpu,  #: List[bool],
             release_times,  #: List[int],
             absolute_deadlines,  #: List[int],
             expected_runtimes,  #: List[int],
             dependency_matrix,
             pinned_tasks,  #: List[int],
             num_tasks,  #: int,
             num_gpus,  #: int,
             num_cpus,  #: int,
             optimize=True,
             dump=False,  # indicates if we shud log the ILP call
             outpath=None,
             dump_nx=False)

        result = list(zip(start_times, placements))
        return sched_runtime, result
