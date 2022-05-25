import random
import uuid
from typing import Mapping, Optional, Sequence

from workload.graph import Graph


class Job(object):
    """A `Job` represents a particular Operator in an ERDOS AV pipeline.

    A `Job` is a static entity and along with other `Job`s creates a `JobGraph`
    which represents the static structure of the computation.

    The `Job`s release a set of `Task`s that are runtime entities and must be
    scheduled by a `Scheduler`.

    Args:
        name: The name of the ERDOS operator that corresponds to this Job.
        runtime: The expected runtime of the tasks created from this Job.
        pipelined (`bool`): True if job's tasks from different timestamps can run
            in parallel.
    """

    def __init__(self, name: str, runtime: int, pipelined: bool = False):
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._runtime = runtime
        self._pipelined = pipelined

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def runtime(self):
        return self._runtime

    @property
    def pipelined(self):
        return self._pipelined

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return f"Job(name={self.name}, id={self.id}, pipelined={self.pipelined})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)


class JobGraph(Graph[Job]):
    """A `JobGraph` represents a directed Operator graph that makes up an
    ERDOS AV pipeline.

    A `JobGraph` is a static entity that defines the relationship between the
    different `Job`s of the application.

    Args:
        jobs: A Mapping from a set of `Job`s to their children that needs to be
            initialized into a `JobGraph`.
    """

    def __init__(
        self,
        jobs: Optional[Mapping[Job, Sequence[Job]]] = {},
    ):
        super().__init__(jobs)

    def add_job(self, job: Job, children: Optional[Sequence[Job]] = []):
        """Adds the job to the graph along with the given children.

        Args:
            job (`Job`): The job to be added to the graph.
            children (`Sequence[Job]`): The children of this job, if any.
        """
        self.add_node(job, *children)

    def pipeline_source_operators(self):
        """Ensures that the source operators pipeline tasks."""
        for job in self.get_sources():
            job._pipelined = True
