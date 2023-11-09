import abc
from typing import Generator

from workload.workload import Workload


class WorkloadLoaderV2(abc.ABC):
    """
    Abstract base class for loading Workload.
    """

    @abc.abstractmethod
    def workloads(self) -> Generator[Workload, None, None]:
        """
        A generator function to load Workload.
        """
        pass

