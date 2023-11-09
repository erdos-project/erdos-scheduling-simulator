import abc
from typing import Generator

from workload.workload import Workload


class WorkloadLoaderDynamic(abc.ABC):
    """
    Abstract base class for workload loaders that can load workload dynamically
    instead of all at once.
    """

    @abc.abstractmethod
    def get_workloads(self) -> Generator[Workload, None, None]:
        """
        A generator function to load Workload.
        """
        pass

