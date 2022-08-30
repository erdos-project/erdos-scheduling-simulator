from typing import Optional, Sequence, Tuple

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Task, Workload


class BranchPredictionScheduler(BaseScheduler):
    """Implements a branch prediction scheduling algorithm for the Simulator.

    Args:
        preemptive (`bool`): If `True`, the scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the simulator (in us). If -1,
            the scheduler returns the actual runtime.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        super(BranchPredictionScheduler, self).__init__(
            preemptive=preemptive, runtime=runtime, _flags=_flags
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Tuple[EventTime, Sequence[Tuple[Task, str, EventTime]]]:
        pass
