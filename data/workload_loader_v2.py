import abc
from typing import Generator

from workload.workload import Workload


class WorkloadLoaderV2(abc.ABC):
    """
    Abstract base class for loading jobs in configurable batch sizes.
    """

    def __init__(self, batch_size: int = 0):
        """
        :param batch_size: The batch size to use when loading jobs. If 0,
            all jobs will be loaded at once.
        """
        self._batch_size = batch_size

    @abc.abstractmethod
    def workloads(self) -> Generator[Workload, None, None]:
        """
        A generator function to load jobs in batches.

        Must be implemented by subclasses.
        """
        pass

