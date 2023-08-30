try:
    from erdos_scheduler_cpp import ERDOSScheduler as ERDOSchedulerCPP
except ModuleNotFoundError:
    print("ERDOSSchedulerCPP was not found. Please compile the C++ implementation.")


from schedulers import BaseScheduler
from utils import EventTime


class ERDOSScheduler(BaseScheduler):
    def __init__(self, preemptive: bool, runtime: EventTime) -> None:
        self._scheduler = ERDOSchedulerCPP(preemptive, runtime)
