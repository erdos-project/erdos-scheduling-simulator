import os
import time
from copy import copy, deepcopy
from operator import attrgetter
from typing import Optional

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Placement, Placements, Workload


class EDFScheduler(BaseScheduler):
    """Implements the EDF scheduling algorithm for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the EDF scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        super(EDFScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        """Implements the BaseScheduler's schedule() method using the EDF
        algorithm for scheduling the released tasks across the worker_pools.
        """
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            time=sim_time,
            preemption=self.preemptive,
            worker_pools=worker_pools,
        )

        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = copy(worker_pools)

        for worker_pool in schedulable_worker_pools.worker_pools:
            self._logger.debug(
                f"[{sim_time}] The state of {worker_pool} is:{os.linesep}"
                f"{os.linesep.join(worker_pool.get_utilization())}"
            )

        # Sort the tasks according to their deadlines, and place them on the
        # worker pools.
        start_time = time.time()
        ordered_tasks = list(sorted(tasks_to_be_scheduled, key=attrgetter("deadline")))

        task_descriptions = [
            f"{task.unique_name} ({task.deadline})" for task in ordered_tasks
        ]
        self._logger.debug(
            f"[{sim_time}] The order of the tasks is {task_descriptions}."
        )

        # Run the scheduling loop.
        # TODO (Sukrit): This loop may require spurious migrations of tasks
        # by preempting them from one pool, and assigning them to another.
        # We should ensure that we check if the worker pool already has running
        # tasks of lower priority, and only preempt the lowest priority one if
        # need be.
        placements = []
        for task in ordered_tasks:
            self._logger.debug(
                f"[{sim_time}] {self.__class__.__name__} trying to schedule {task} "
                f"with the resource requirements {task.resource_requirements}."
            )
            is_task_placed = False
            for worker_pool in schedulable_worker_pools.worker_pools:
                if worker_pool.can_accomodate_task(task):
                    worker_pool.place_task(task)
                    is_task_placed = True
                    placements.append(
                        Placement(
                            task=task,
                            worker_pool_id=worker_pool.id,
                            placement_time=sim_time,
                        )
                    )
                    self._logger.debug(
                        f"[{sim_time}] Placed {task} on Worker Pool ({worker_pool.id})"
                        f" to be started at {sim_time}."
                    )
                    break

            if is_task_placed:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    self._logger.debug(
                        f"[{sim_time}] The state of {worker_pool} is:{os.linesep}"
                        f"{os.linesep.join(worker_pool.get_utilization())}"
                    )
            else:
                self._logger.debug(
                    f"[{sim_time}] Failed to place {task} because no worker pool "
                    f"could accomodate the resource requirements."
                )
                placements.append(Placement(task=task))

        end_time = time.time()
        if self.runtime == EventTime(time=-1, unit=EventTime.Unit.US):
            return Placements(
                runtime=EventTime(
                    int((end_time - start_time) * 1e6), EventTime.Unit.US
                ),
                placements=placements,
            )
        else:
            return Placements(runtime=self.runtime, placements=placements)
