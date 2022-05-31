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

    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        _flags: Optional["absl.flags"] = None,
    ):
        assert not preemptive, "FIFO scheduler is not preemptive"
        super(FIFOScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )

    def schedule(
        self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools
    ) -> (int, Sequence[Tuple[Task, str]]):
        tasks = task_graph.get_schedulable_tasks(
            sim_time, 0, self.preemptive, worker_pools
        )
        # Create a virtual WorkerPool set to try scheduling decisions on.
        schedulable_worker_pools = copy(worker_pools)

        start_time = time.time()
        # Sort the tasks according to their release times, and place them on
        # the worker pools.
        ordered_tasks = list(sorted(tasks, key=attrgetter("release_time")))

        placements = []
        for task in ordered_tasks:
            task_placed = False
            for worker_pool in schedulable_worker_pools.worker_pools:
                if worker_pool.can_acommodate_task(task):
                    worker_pool.place_task(task)
                    task_placed = True
                    placements.append((task, worker_pool.id, sim_time))
                    break
            if not task_placed:
                placements.append((task, None, None))

        end_time = time.time()
        if self.runtime == -1:
            return int((end_time - start_time) * 1000000), placements
        else:
            return self.runtime, placements
