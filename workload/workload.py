from typing import Mapping, Optional

from .jobs import JobGraph
from .tasks import TaskGraph


class Workload(object):
    """A representation of the Workload that is to be simulated.

    The Workload can be continually generated from the set of JobGraphs provided
    to it, or might have a single TaskGraph provided to it that is released at
    the beginning of the simulation.
    """

    def __init__(
        self, job_graphs: Mapping[str, JobGraph], task_graphs: Mapping[str, TaskGraph]
    ) -> None:
        self._job_graphs = job_graphs

    @staticmethod
    def from_job_graphs(job_graphs: Mapping[str, JobGraph]) -> "Workload":
        """Creates a Workload from the given JobGraphs.

        Args:
            job_graphs: A mapping from the name of the application to its JobGraph.

        """
        return Workload(job_graphs=job_graphs, task_graphs=dict())

    def get_job_graph(self, name: str) -> Optional[JobGraph]:
        """Retrieves the JobGraph for the given application, if present.

        Args:
            name (`str`): The name of the application.

        Returns:
            The `JobGraph` representing the application, if available.
        """
        return self._job_graphs.get(name)
