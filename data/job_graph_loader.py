import abc
from typing import Generator, Sequence
from workload.jobs import JobGraph

from workload.workload import Workload


class JobGraphLoader(abc.ABC):
    """
    Abstract base class for workload loaders that can load workload dynamically
    instead of all at once.
    """

    @abc.abstractmethod
    def get_next_jobs(self, start_time_offset: int = 0) -> Sequence[JobGraph]:
        """
        A generator function to load Workload.
        """
        pass

