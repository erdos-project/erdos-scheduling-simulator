import abc
from typing import Generator

from workload.workload import Workload


class WorkloadLoaderDynamic(abc.ABC):
    """
    Abstract base class for workload loaders that can load workload dynamically
    instead of all at once.
    """

    @abc.abstractmethod
    def get_next_workload(self, start_time_offset: int = 0) -> Workload:
        """
        A generator function to load Workload.
        """
        pass

