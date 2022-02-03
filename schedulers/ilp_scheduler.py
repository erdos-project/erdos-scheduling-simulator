from typing import List
from schedulers.base_scheduler import BaseScheduler
from typing import Sequence, Tuple
from copy import copy, deepcopy

from workload import Task, TaskGraph, Resource
from workers import WorkerPool


def verify_schedule(start_times, placements, resource_requirements, release_times,
                    absolute_deadlines, expected_runtimes, dependency_matrix,
                    num_resources):
    # check release times
    assert all([s >= r for s, r in zip(start_times, release_times)
                ]), "not_valid_release_times"
    assert all([(not need_gpu or (p >= num_cpus))
                for need_gpu, p in zip(needs_gpu, placements)
                ]), "not_valid_placement"
    assert all([
        (d >= s + e)
        for d, e, s in zip(absolute_deadlines, expected_runtimes, start_times)
    ]), "doesn't finish before deadline"
    for i, row_i in enumerate(dependency_matrix):
        for j, column_j in enumerate(row_i):
            if i != j and column_j:
                assert start_times[i] + expected_runtimes[i] <= start_times[
                    j], f"not_valid_dependency{i}->{j}"
    placed_tasks = [
        (p, s, s + e)
        for p, s, e in zip(placements, start_times, expected_runtimes)
    ]
    placed_tasks.sort()
    for t1, t2 in zip(placed_tasks, placed_tasks[1:]):
        if t1[0] == t2[0]:
            print(t1, t2)
            assert t1[2] <= t2[1], f"overlapping_tasks_on_{t1[0]}"


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

    def __init__(self,
                 sched_solver: ILPScheduler,
                 preemptive: bool = False,
                 runtime: float = -1.0):
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
        
        # unique list of resource names -- not relying on set stability
        resource_names = list({ r.name  for wp in schedulable_worker_pools for r in wp.resources._resource_vector.keys()})

        r_maps = { r_name:
            {
            wp.id:
            wp.resources.get_available_quantity(Resource(name=r_name,
                                                         _id="any"))
            for wp in schedulable_worker_pools
            }
            for r_name in resource_names
        }
        
        num_resources = [
            sum(r_maps[r_name].values())
            for r_name in resource_names
        ]

        estimated_scheduling_overhead = 0
        num_tasks = len(tasks_to_be_scheduled)
        absolute_deadlines = [task.deadline for task in tasks_to_be_scheduled]
        release_times = [estimated_scheduling_overhead] * num_tasks
        pinned_tasks = [None] * num_tasks

        expected_runtimes = [
            task.remaining_time for task in tasks_to_be_scheduled
        ]

        resource_requirements = [
            [
                task.resource_requirements < Resource(name=r_name, _id="any")
                for r_name in resource_names
            ]
            for task in tasks_to_be_scheduled
        ]

        # TODO (Justin) : This doesn't account for the dependencies
        # between tasks.
        dependency_matrix = [[False] * num_tasks] * num_tasks

        (start_times,
         placements), opt_value, sched_runtime = self.sched_solver.schedule(
             resource_requirements, #: List<tasks>[List<uniq_resources>[bool]]
             release_times,  #: List<tasks>[int],
             absolute_deadlines,  #: List<tasks>[int],
             expected_runtimes,  #: List<tasks>[int],
             dependency_matrix, #: List<tasks>[List<tasks>[bool]],
             pinned_tasks,  #: List<tasks>[int<total_num_resources>],
             num_tasks,  #: int,
             [num_resources[r_name] for r_name in resource_names], #: List<uniq_resources>[int],
             optimize=True,
             log_dir=None,
             verbose=False)

        if opt_value is None:  # Doesn't handle loadshedding
            return (sched_runtime, [])

        verify_schedule(start_times, placements, resource_requirements, release_times,
                        absolute_deadlines, expected_runtimes,
                        dependency_matrix, num_resources)

        resource_map = {
            r_name:
            [[wp_id] * r_maps[wp_id] for wp_id in r_maps[r_name].keys()]
            for r_name in r_maps.keys()
        }

        resource_map = {
            r_name:
            [j for sub in resource_map[r_name] for j in sub] # flatten
            for r_name in resource_map.keys()
        }
        # TODO: this doesn't actually handle the resource map ordering

        placements = [(t, resource_map[p - 1])
                      for t, p in zip(tasks_to_be_scheduled, placements)]

        return (sched_runtime, placements)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime

    def _schedule(self, tasks, task_graph,
                  worker_pools) -> Tuple[float, Sequence[Tuple[int, int]]]:
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
             log_dir=None)

        result = list(zip(start_times, placements))
        return sched_runtime, result
