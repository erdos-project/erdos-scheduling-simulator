from typing import List
from schedulers.base_scheduler import BaseScheduler
from typing import Sequence, Mapping, Tuple, Optional
from copy import copy, deepcopy

from workload import Task, TaskGraph, Resource
from workers import WorkerPool

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
    def __init__(self, sched_solver: ILPScheduler, preemptive: bool = False, runtime: float = -1.0):
        self.sched_solver = sched_solver()
        self._preemptive = preemptive
        self._runtime = runtime

    def schedule(self, sim_time: float, released_tasks: Sequence[Task],
                 task_graph: TaskGraph, worker_pools: Sequence[WorkerPool])\
            -> (float, Sequence[Tuple[Task, str]]):
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        if self.preemptive:
            # Collect all the currently placed tasks on the WorkerPool, along
            # with the set of released tasks.
            # TODO (Sukrit): Should we check if they are currently running?
            tasks_to_be_scheduled = [task for task in released_tasks]
            for worker_pool in worker_pools:
                tasks_to_be_scheduled.extend(worker_pool.get_placed_tasks())

            # Restart the state of the WorkerPool.
            schedulable_worker_pools = [deepcopy(w) for w in worker_pools]
        else:
            # Collect the currently released tasks.
            tasks_to_be_scheduled = [task for task in released_tasks]

            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = [copy(w) for w in worker_pools]
        
        import IPython; IPython.embed()
        
        cpu_map = { wp.id : wp.get_available_quantity(Resource(name="CPU", _id="any")) for wp in schedulable_worker_pools}
        gpu_map = { wp.id : wp.get_available_quantity(Resource(name="GPU", _id="any")) for wp in schedulable_worker_pools}
        
        num_cpus = sum(cpu_map.values())
        num_gpus = sum(gpu_map.values()) 

        estimated_scheduling_overhead = 0
        num_tasks = len (released_tasks)
        absolute_deadlines = [task.deadline for task in released_tasks]
        release_times = [estimated_scheduling_overhead] * num_tasks
        pinned_tasks = [None] * num_tasks

        expected_runtimes = [task._expected_runtime for task in released_tasks]
        gpu_resource_requirement = [task.resource_requirements.get_available_quantity(Resource(name="GPU", _id="any")) for task in released_tasks]
        needs_gpu = [r > 0 for r in gpu_resource_requirement]
        dependency_matrix = [[False] * num_tasks] * num_tasks

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

        cpu_map = [ [wp_id]*cpu_map[wp_id] for wp_id in cpu_map.keys()]
        cpu_map = [j for sub in cpu_map for j in sub]

        gpu_map = [ [wp_id]*gpu_map[wp_id] for wp_id in gpu_map.keys()]
        gpu_map = [j for sub in gpu_map for j in sub]

        resource_map = gpu_map + cpu_map

        placements = [resource_map[p] for p in placements]

        return (sched_runtime, placements)

        
    
    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime


    def _schedule(self, tasks, task_graph, worker_pools)-> Tuple[float,Sequence[Tuple[int,int]]]:
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
