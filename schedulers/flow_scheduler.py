import time
from typing import List, Mapping, Optional, Set, Tuple

import absl  # noqa: F401
import flowlessly_py

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Placement, Placements, Resource, Task, TaskGraph, Workload


class FlowScheduler(BaseScheduler):
    """Implements a scheduler based on min cost max flow algorithm.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        if preemptive:
            raise ValueError("FlowScheduler does not support preemption.")
        super(FlowScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled: List[Task] = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
            release_taskgraphs=self.release_taskgraphs,
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks to be scheduled. These tasks along with their deadlines were: "
            f"{[f'{t.unique_name} ({t.deadline})' for t in tasks_to_be_scheduled]}"
        )

        scheduler_start_time = time.time()
        placements = []
        if len(tasks_to_be_scheduled) > 0:
            # TODO
            pass

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime if self.runtime == EventTime.invalid() else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )