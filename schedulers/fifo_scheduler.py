import time
from copy import copy
from operator import attrgetter
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from workers import WorkerPools
from workload import Task, TaskGraph


class FIFOScheduler(BaseScheduler):
    """Implements the FIFO scheduling algorithm.

    Args:
        runtime (`int`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(self,
                 preemptive: bool = False,
                 runtime: int = -1,
                 _flags: Optional['absl.flags'] = None):
        assert not preemptive, "FIFO scheduler is not preemptive"
        self._preemptive = preemptive
        self._runtime = runtime
        # The scheduler only places tasks that have been released. Hence,
        # scheduling horizon is 0.
        self._scheduling_horizon = 0

    def schedule(
            self, sim_time: int, task_graph: TaskGraph,
            worker_pools: WorkerPools) -> (int, Sequence[Tuple[Task, str]]):
        tasks = task_graph.get_schedulable_tasks(sim_time, 0, self.preemptive,
                                                 worker_pools)
        # Create a virtual WorkerPool set to try scheduling decisions on.
        schedulable_worker_pools = copy(worker_pools)

        start_time = time.time()
        # Sort the tasks according to their release times, and place them on
        # the worker pools.
        ordered_tasks = list(sorted(tasks, key=attrgetter('release_time')))

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
        self._runtime = int((end_time - start_time) *
                            1000000) if self.runtime == -1 else self.runtime

        return self.runtime, placements

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime

    @property
    def scheduling_horizon(self):
        return self._scheduling_horizon
