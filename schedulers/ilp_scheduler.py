from typing import List
from schedulers.base_scheduler import BaseScheduler
from typing import Optional, Sequence, Tuple
from copy import copy, deepcopy

from workload import Task, TaskGraph, Resource, Resources
from workers import WorkerPool


def verify_schedule(start_times, placements, resource_requirements, tasks,
                    dependency_matrix, num_resources):
    # Check if each task's start time is greater than its release time.
    assert all([
        start_time >= task.release_time
        for start_time, task in zip(start_times, tasks)
    ]), "not_valid_release_times"

    num_resources_ub = [
        sum(num_resources[0:i + 1]) for i in range(len(num_resources))
    ]
    num_resources_lb = [0] + [
        sum(num_resources[0:i + 1]) for i in range(len(num_resources) - 1)
    ]
    assert all([
        all([
            p < num_resources_ub[i] and num_resources_lb[i] <= p
            if resource_req else True
            for i, resource_req in enumerate(resource_req_arr)
        ]) for resource_req_arr, p in zip(resource_requirements, placements)
    ]), "not_valid_placement"

    # Check if all tasks finished before the deadline.
    assert all([(task.deadline >= start_time + task.runtime)
                for task, start_time in zip(tasks, start_times)
                ]), "doesn't finish before deadline"

    # Check if the task dependencies were satisfied.
    for i, row_i in enumerate(dependency_matrix):
        for j, col_j in enumerate(row_i):
            if i != j and col_j:
                assert start_times[i] + tasks[i].runtime <= start_times[
                    j], f"not_valid_dependency{i}->{j}"

    # Check if tasks overlapped on a resource.
    placed_tasks = [
        (placement, start_time, start_time + task.runtime)
        for placement, start_time, task in zip(placements, start_times, tasks)
    ]
    placed_tasks.sort()
    for t1, t2 in zip(placed_tasks, placed_tasks[1:]):
        if t1[0] == t2[0]:
            assert t1[2] <= t2[1], f"overlapping_tasks_on_{t1[0]}"


def compute_slack_cost(placement, expected_runtime, absolute_deadlines):
    slacks = [
        d - e - p
        for p, e, d in zip(placement, expected_runtime, absolute_deadlines)
    ]
    return sum(slacks)


class ILPScheduler(BaseScheduler):

    def __init__(self,
                 sched_solver,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 _flags: Optional['absl.flags'] = None):
        self.sched_solver = sched_solver(preemptive, runtime, _flags)
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
            tasks = [task for task in released_tasks]
            for worker_pool in worker_pools:
                tasks.extend(worker_pool.get_placed_tasks())

            # Restart the state of the WorkerPool.
            schedulable_worker_pools = [deepcopy(w) for w in worker_pools]
        else:
            # Collect the currently released tasks.
            tasks = [task for task in released_tasks]

            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = [copy(w) for w in worker_pools]

        # unique list of resource names -- not relying on set stability
        resource_names = list({
            r.name
            for wp in schedulable_worker_pools
            for r in wp.resources._resource_vector.keys()
        })
        # uniquify scrambles the order
        resource_names.sort()

        # {resource_type : {key = pool ID : value = number of resource_type}}
        r_maps = {
            r_name: {
                wp.id: wp.resources.get_available_quantity(
                    Resource(name=r_name, _id="any"))
                for wp in schedulable_worker_pools
            }
            for r_name in resource_names
        }

        # {resource_type : {key = pool ID :
        #          value = number of resource_type}}
        num_resources = [
            sum(r_maps[r_name].values()) for r_name in resource_names
        ]  # [number of resources of each type] ordered by resource_names

        resource_requirements = [[
            task.resource_requirements > Resources(
                {Resource(name=r_name, _id="any"): 1})
            for r_name in resource_names
        ] for task in tasks]

        # [[true iff task fits on resource type r for r in uniq_resource ]
        #  for each task]

        # TODO(Justin): This doesn't account for the task dependencies.
        dependency_matrix = [[False] * len(tasks)] * len(tasks)

        (start_times,
         placements), opt_value, sched_runtime = self.sched_solver.schedule(
             resource_requirements,  #: List<tasks>[List<uniq_resources>[bool]]
             tasks,
             dependency_matrix,  #: List<tasks>[List<tasks>[bool]],
             num_resources)  #: List<uniq_resources>[int])

        if opt_value is None:
            # Doesn't handle loadshedding.
            return (sched_runtime, [])

        verify_schedule(start_times, placements, resource_requirements, tasks,
                        dependency_matrix, num_resources)

        # {resource_type : List<unique_wp_id>[
        #    List<one-id-per-quantity>[pool ID]]}
        resource_map = {
            r_name: [[wp_id] * r_maps[r_name][wp_id]
                     for wp_id in r_maps[r_name].keys()]
            for r_name in r_maps.keys()
        }

        # {resource_type:
        #    List<unique_wp_id>[List<one-id-per-quantity>[pool ID]]}
        resource_map = [
            [j for sub in resource_map[r_name]
             for j in sub]  # flatten along wps
            for r_name in resource_names
        ]  # List<resource_names>[List<wp_ids_by_quantity>[pool ID]]

        # Flatten again along resource type.
        resource_map = [j for sub in resource_map for j in sub]

        placements = [(t, resource_map[p]) for t, p in zip(tasks, placements)]

        return (sched_runtime, placements)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime
