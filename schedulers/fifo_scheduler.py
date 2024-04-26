import os
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
        enforce_deadlines: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        assert not preemptive, "FIFO scheduler is not preemptive"
        super(FIFOScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            enforce_deadlines=enforce_deadlines,
            _flags=_flags,
        )
        if _flags is not None:
            if _flags.release_taskgraphs:
                raise ValueError("FIFOScheduler does not support taskgraphs.")

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        tasks = workload.get_schedulable_tasks(
            time=sim_time,
            preemption=self.preemptive,
            worker_pools=worker_pools,
            scheduler_is_task_type=True,
        )
        # Create a virtual WorkerPool set to try scheduling decisions on.
        # schedulable_worker_pools = copy(worker_pools)
        schedulable_worker_pools = worker_pools

        start_time = time.time()
        # Sort the tasks according to their release times, and place them on
        # the worker pools.
        ordered_tasks = list(sorted(tasks, key=attrgetter("release_time")))

        # TODO: (DG) Check Alind's changes in EDF scheduler
        # and replicate code for dropping tasks?

        placements = []
        for task in ordered_tasks:
            self._logger.debug(
                f"[{sim_time.time}] FIFOScheduler trying to schedule {task} with the "
                f"available execution strategies: "
                f"{task.available_execution_strategies}."
            )

            # If we are enforcing deadlines, and the Task is past its deadline, then
            # we should create a cancellation for it. This is only applicable if the
            # user wants the tasks that cannot meet their deadline to be dropped.
            if (
                self.enforce_deadlines
                and task.deadline
                < sim_time
                + task.available_execution_strategies.get_fastest_strategy().runtime
            ):
                placements.append(Placement.create_task_cancellation(task=task))
                self._logger.debug(
                    "[%s] Task %s has a deadline of %s, which has been missed. "
                    "Cancelling the task.",
                    sim_time.time,
                    task,
                    task.deadline.time,
                )
                continue

            # Try to place the task on the worker pools.
            is_task_placed = False
            for execution_strategy in task.available_execution_strategies:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    if worker_pool.can_accomodate_strategy(execution_strategy):
                        worker_pool.place_task(
                            task, execution_strategy=execution_strategy
                        )
                        is_task_placed = True
                        placements.append(
                            Placement.create_task_placement(
                                task=task,
                                placement_time=sim_time,
                                worker_pool_id=worker_pool.id,
                                execution_strategy=execution_strategy,
                            )
                        )
                        self._logger.debug(
                            f"[{sim_time.time}] Placed {task} on Worker Pool "
                            f"({worker_pool.id}) to be started at {sim_time} with the "
                            f"execution strategy: {execution_strategy}."
                        )
                        break
                if is_task_placed:
                    break

            if is_task_placed:
                for worker_pool in schedulable_worker_pools.worker_pools:
                    self._logger.debug(
                        f"[{sim_time.time}] The state of {worker_pool} is:{os.linesep}"
                        f"{os.linesep.join(worker_pool.get_utilization())}"
                    )
            else:
                self._logger.debug(
                    "[%s] Failed to place %s because no worker pool "
                    "could accomodate the resource requirements.",
                    sim_time.time,
                    task,
                )
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
