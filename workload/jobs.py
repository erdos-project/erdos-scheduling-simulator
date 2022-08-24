import random
import uuid
from typing import Mapping, Optional, Sequence, Tuple

from utils import EventTime
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
        conditional (`bool`): True if only some of the job's childrens are invoked
            upon the job's completion instead of all of them.
        terminal (`bool`): True if the job is a terminal job of the conditional
            node (can also be constructed using `create_conditional_pair` method).
    """

    def __init__(
        self,
        name: str,
        runtime: EventTime,
        pipelined: bool = False,
        conditional: bool = False,
        terminal: bool = False,
    ) -> None:
        if type(runtime) != EventTime:
            raise ValueError(f"Invalid type received for runtime: {type(runtime)}")
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._runtime = runtime
        self._pipelined = pipelined
        self._conditional = conditional
        self._terminal = terminal

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

    @property
    def conditional(self):
        return self._conditional

    @property
    def terminal(self):
        return self._terminal

    @staticmethod
    def create_conditional_pair(name: str) -> Tuple["Job", "Job"]:
        """Create a conditional pair with zero runtime on both ends.

        This method can be used to construct a pair of if-else constructs
        in the `JobGraph`.

        Args:
            name: A name to give to the Conditional statement.
        """
        conditional_begin = Job(
            f"{name}_conditional", runtime=EventTime.zero(), conditional=True
        )
        conditional_end = Job(
            f"{name}_terminal", runtime=EventTime.zero(), terminal=True
        )
        return conditional_begin, conditional_end

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return (
            f"Job(name={self.name}, id={self.id}, "
            f"pipelined={self.pipelined}, conditional={self.conditional})"
        )

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
        completion_time: The completion time to assign to this `JobGraph`. If `None`,
            the completion time will be computed as the sum of the runtime of the
            longest path in the graph.
    """

    def __init__(
        self,
        jobs: Optional[Mapping[Job, Sequence[Job]]] = {},
        completion_time: Optional[EventTime] = None,
    ):
        super().__init__(jobs)
        self._completion_time = (
            completion_time
            if completion_time or len(self) == 0
            else sum(
                (job.runtime for job in self.get_longest_path()),
                start=EventTime(0, EventTime.Unit.US),
            )
        )

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

    @property
    def completion_time(self):
        if not self._completion_time and len(self) != 0:
            self._completion_time = sum(
                (job.runtime for job in self.get_longest_path()),
                start=EventTime(0, EventTime.Unit.US),
            )
        return self._completion_time
