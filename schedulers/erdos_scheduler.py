from erdos_scheduler_cpp import ERDOSScheduler as ERDOSchedulerCPP

from schedulers import BaseScheduler
from utils import EventTime


class ERDOSScheduler(BaseScheduler):
    def __init__(self, preemptive: bool, runtime: EventTime) -> None:
        self._scheduler = ERDOSchedulerCPP(preemptive, runtime)
