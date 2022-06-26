import time
from copy import copy, deepcopy
from functools import partial
from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Task, TaskGraph


class LSFScheduler(BaseScheduler):
    """Implements the Least Slack First (LSF) scheduling algorithm for the
    Simulator.

    Args:
        preemptive (`bool`): If `true`, the LSF scheduler can preempt the tasks
            that are currently running.
        runtime (`int`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        super(LSFScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )

    def schedule(
        self, sim_time: EventTime, task_graph: TaskGraph, worker_pools: WorkerPools
    ) -> (EventTime, Sequence[Tuple[Task, str, EventTime]]):
        """Implements the BaseScheduler's schedule() method using the LSF
        algorithm for scheduling the released tasks across the worker_pools.
        """
        # Create the tasks to be scheduled, along with the state of the
        # WorkerPool to schedule them on based on preemptive or non-preemptive
        tasks_to_be_scheduled = task_graph.get_schedulable_tasks(
            sim_time, EventTime(0, EventTime.Unit.US), self.preemptive, worker_pools
        )
        if self.preemptive:
            # Restart the state of the WorkerPool.
            schedulable_worker_pools = deepcopy(worker_pools)
        else:
            # Create a virtual WorkerPool set to try scheduling decisions on.
            schedulable_worker_pools = copy(worker_pools)

        # Sort the tasks according to their slacks, and place them on the
        # worker pools.
        start_time = time.time()
        ordered_tasks = list(
            sorted(tasks_to_be_scheduled, key=partial(self.slack, sim_time))
        )

        # Run the scheduling loop.
        placements = []
        for task in ordered_tasks:
            is_task_placed = False
            for worker_pool in schedulable_worker_pools.worker_pools:
                if worker_pool.can_accomodate_task(task):
                    worker_pool.place_task(task)
                    is_task_placed = True
                    placements.append((task, worker_pool.id, sim_time))
                    break

            if not is_task_placed:
                placements.append((task, None, None))

        end_time = time.time()
        if self.runtime == EventTime(-1, EventTime.Unit.US):
            return (
                EventTime(int((end_time - start_time) * 1e6), EventTime.Unit.US),
                placements,
            )
        else:
            return self.runtime, placements

    def slack(self, sim_time: int, task: Task) -> int:
        """Defines the Slack used by the scheduler to order the events.

        Args:
            sim_time (`int`): The time in us at which the scheduler was
                invoked.
            task (`Task`): The Task to calculate the slack of.

        Returns:
            A `int` value depicting the slack of the task.
        """
        return task.deadline - sim_time - task.remaining_time
