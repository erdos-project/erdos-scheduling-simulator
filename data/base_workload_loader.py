from typing import Optional

from workload import Workload


class BaseWorkloadLoader(object):
    """Base class for Workload loaders that the Simulator uses to retrieve new
    TaskGraphs into the system."""

    def get_next_workload(self) -> Optional[Workload]:
        """Retrieve the updated Workload from the Loader.

        Returns:
            An optional Workload. If the return is None, then no new TaskGraphs
            were released into the Workload, and the Simulator should finish
            the remainder of the TaskGraphs in the current Workload and exit.
        """
        raise NotImplementedError(
            f"The WorkloadLoader of type {type(self)} does not implement "
            f"get_next_workload"
        )
