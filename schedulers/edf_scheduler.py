import time
from operator import attrgetter
from copy import copy, deepcopy
from typing import Sequence, Tuple

from schedulers import BaseScheduler
from workload import Task, TaskGraph
from workers import WorkerPool


class EDFScheduler(BaseScheduler):
    """Implements the EDF scheduling algorithm for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the EDF scheduler can preempt the tasks
            that are currently running.
        runtime (`float`): The runtime to return to the simulator. If -1, the
            scheduler returns the actual runtime.
    """

    def __init__(self, preemptive: bool = False, runtime: float = -1.0):
        self._preemptive = preemptive
        self._runtime = runtime

    def schedule(self, sim_time: float, released_tasks: Sequence[Task],
                 task_graph: TaskGraph, worker_pools: Sequence[WorkerPool])\
            -> (float, Sequence[Tuple[Task, str]]):
        """Implements the BaseScheduler's schedule() method using the EDF
        algorithm for scheduling the given released_tasks across the
        worker_pools.
        """
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

        # Sort the tasks according to their deadlines, and place them on the
        # worker pools.
        start_time = time.time()
        ordered_tasks = list(
            sorted(tasks_to_be_scheduled, key=attrgetter('deadline')))

        # Run the scheduling loop.
        # TODO (Sukrit): This loop may require spurious migrations of tasks
        # by preempting them from one pool, and assigning them to another.
        # We should ensure that we check if the worker pool already has running
        # tasks of lower priority, and only preempt the lowest priority one if
        # need be.
        placements = []
        for task in ordered_tasks:
            is_task_placed = False
            for worker_pool in schedulable_worker_pools:
                if worker_pool.can_accomodate_task(task):
                    worker_pool.place_task(task)
                    is_task_placed = True
                    placements.append((task, worker_pool.id))
                    break

            if not is_task_placed:
                placements.append((task, None))

        end_time = time.time()

        return (end_time - start_time if self.runtime == -1 else self.runtime,
                placements)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime
