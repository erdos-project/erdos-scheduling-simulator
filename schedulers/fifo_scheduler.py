import time
from copy import copy
from operator import attrgetter
from typing import Optional

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Placement, Placements, Workload


class FIFOScheduler(BaseScheduler):
    """Implements the FIFO scheduling algorithm.

    Args:
        runtime (`int`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        assert not preemptive, "FIFO scheduler is not preemptive"
        super(FIFOScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        tasks = workload.get_schedulable_tasks(
            time=sim_time, preemption=self.preemptive, worker_pools=worker_pools
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
                if worker_pool.can_accomodate_task(task):
                    worker_pool.place_task(task)
                    task_placed = True
                    placements.append(
                        Placement.create_task_placement(
                            task=task,
                            worker_pool_id=worker_pool.id,
                            placement_time=sim_time,
                        )
                    )
                    break
            if not task_placed:
                placements.append(Placement.create_task_placement(task=task))

        end_time = time.time()

        # Compute and return the Placements object.
        scheduler_runtime = EventTime(
            int((end_time - start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime
            if self.runtime == EventTime(-1, EventTime.Unit.US)
            else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )
