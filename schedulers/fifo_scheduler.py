import time
from operator import attrgetter
from copy import copy
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from workload import Task, TaskGraph
from workers import WorkerPools


class FIFOScheduler(BaseScheduler):
    """Implements the FIFO scheduling algorithm.

    Args:
        runtime (`float`): The runtime to return to the simulator. If -1, the
            scheduler returns the actual runtime.
    """

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 _flags: Optional['absl.flags'] = None):
        assert not preemptive, "FIFO scheduler is not preemptive"
        self._preemptive = preemptive
        self._runtime = runtime

    def schedule(self, sim_time: float, released_tasks: Sequence[Task],
                 task_graph: TaskGraph, worker_pools: WorkerPools)\
            -> (float, Sequence[Tuple[Task, str]]):
        # Create a virtual WorkerPool set to try scheduling decisions on.
        schedulable_worker_pools = copy(worker_pools)

        start_time = time.time()
        # Sort the tasks according to their release times, and place them on
        # the worker pools.
        ordered_tasks = list(
            sorted(released_tasks, key=attrgetter('release_time')))

        placements = []
        for task in ordered_tasks:
            task_placed = False
            for worker_pool in schedulable_worker_pools._wps:
                if worker_pool.can_acommodate_task(task):
                    worker_pool.place_task(task)
                    task_placed = True
                    placements.append((task, worker_pool.id))
                    break
            if not task_placed:
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
