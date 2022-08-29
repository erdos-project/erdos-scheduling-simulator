import logging
import random
import uuid
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

from utils import EventTime, fuzz_time, setup_logging

from .graph import Graph
from .resources import Resources
from .tasks import Task, TaskGraph


class Job(object):
    """A `Job` represents a particular Operator in an ERDOS AV pipeline.

    A `Job` is a static entity and along with other `Job`s creates a `JobGraph`
    which represents the static structure of the computation.

    The `Job`s release a set of `Task`s that are runtime entities and must be
    scheduled by a `Scheduler`.

    Args:
        name: The name of the ERDOS operator that corresponds to this Job.
        runtime: The expected runtime of the tasks created from this Job.
        resource_requirements: A list of Resources that each Task may choose from.
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
        resource_requirements: Optional[Sequence[Resources]] = None,
        pipelined: bool = False,
        conditional: bool = False,
        terminal: bool = False,
    ) -> None:
        if type(runtime) != EventTime:
            raise ValueError(f"Invalid type received for runtime: {type(runtime)}")
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        if resource_requirements is None:
            resource_requirements = [Resources()]
        self._resource_requirements = resource_requirements
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
    def resource_requirements(self):
        return self._resource_requirements

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
            f"Job(name={self.name}, id={self.id}, pipelined={self.pipelined}, "
            f"conditional={self.conditional}, terminal={self.terminal})"
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

    class ReleasePolicyType(Enum):
        """Represents the different release policies supported by a JobGraph."""

        PERIODIC = 1

    class ReleasePolicy(object):
        def __init__(
            self,
            policy_type: "ReleasePolicyType",  # noqa: F821
            period: EventTime = EventTime(100, EventTime.Unit.US),
        ) -> None:
            self._policy_type = policy_type
            self._period = period

        @property
        def policy_type(self) -> "ReleasePolicyType":  # noqa: F821
            return self._policy_type

        @property
        def period(self) -> EventTime:
            return self._period

    def __init__(
        self,
        name: str,
        jobs: Optional[Mapping[Job, Sequence[Job]]] = {},
        release_policy: Optional["ReleasePolicy"] = None,
        completion_time: Optional[EventTime] = None,
    ):
        super().__init__(jobs)
        self._name = name
        self._release_policy = release_policy
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

    def generate_task_graphs(
        self,
        completion_time: EventTime,
        start_time: EventTime = EventTime(0, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ) -> Mapping[str, TaskGraph]:
        """Generates the task graphs under the defined `release_policy`
        and the `completion_time`.

        Args:
            completion_time: The time at which the simulator has to end.
            start_time (default: zero): The time at which the simulator starts.

        Returns:
            A mapping from the name of the `TaskGraph` to the `TaskGraph`.
        """
        if self.release_policy.policy_type == self.ReleasePolicyType.PERIODIC:
            releases = []
            current_release = start_time
            while current_release <= completion_time:
                releases.append(current_release)
                current_release += self.release_policy.period
        else:
            raise NotImplementedError(
                f"The policy {self.release_policy} has not been implemented yet."
            )

        if _flags:
            task_logger = setup_logging(
                name="Task", log_file=_flags.log_file_name, log_level=_flags.log_level
            )
        else:
            task_logger = setup_logging(name="Task")

        # Generate the task graphs for all the releases.
        task_graphs = {}
        for index, release_time in enumerate(releases):
            task_graph_name = f"{self.name}@{index}"
            task_graphs[task_graph_name] = self._generate_task_graph(
                release_time=release_time,
                task_graph_name=task_graph_name,
                timestamp=index,
                task_logger=task_logger,
                _flags=_flags,
            )
        return task_graphs

    def _generate_task_graph(
        self,
        release_time: EventTime,
        task_graph_name: str,
        timestamp: int,
        task_logger: logging.Logger,
        _flags: Optional["absl.flags"] = None,
    ) -> TaskGraph:
        """Generates a TaskGraph from the structure of the `JobGraph` whose
        source operators are released at the provided `release_time`.

        Args:
            `release_time`: The time at which the source operators for the
            `TaskGraph` must be released.
            task_graph_name: The name of the `TaskGraph` being generated.
            timestamp: The timestamp of the `TaskGraph` being generated.

        Returns:
            A `TaskGraph` whose source operators are released at the given time.
        """
        # Retrieve variances from the command line flags.
        if _flags:
            runtime_variance = _flags.runtime_variance
            deadline_variance = (
                _flags.min_deadline_variance,
                _flags.max_deadline_variance,
            )
        else:
            runtime_variance = (0, 0)
            deadline_variance = (0, 0)

        # Generate all the `Task`s from the `Job`s in the graph.
        job_to_task_mapping = {}
        for job in self.breadth_first():
            task_release_time = (
                release_time
                if self.is_source(job)
                else EventTime(-1, EventTime.Unit.US)
            )
            task_runtime = fuzz_time(job.runtime, runtime_variance)
            task_deadline = fuzz_time(
                self.__get_completion_time(start=release_time),
                deadline_variance,
            )
            job_to_task_mapping[job.name] = Task(
                name=job.name,
                task_graph=task_graph_name,
                job=job,
                runtime=task_runtime,
                deadline=task_deadline,
                timestamp=timestamp,
                release_time=task_release_time,
                _logger=task_logger,
            )

        # Generate a TaskGraph from the generated Tasks.
        task_graph_mapping = {}
        for parent, children in self._graph.items():
            task_children = [job_to_task_mapping[child.name] for child in children]
            task_graph_mapping[job_to_task_mapping[parent.name]] = task_children
        return TaskGraph(tasks=task_graph_mapping)

    def __get_completion_time(self, start=EventTime(0, EventTime.Unit.US)) -> EventTime:
        return sum((job.runtime for job in self.get_longest_path()), start=start)

    @property
    def completion_time(self):
        if not self._completion_time and len(self) != 0:
            self._completion_time = self.__get_completion_time()
        return self._completion_time

    @property
    def release_policy(self) -> Optional["ReleasePolicy"]:
        return self._release_policy

    @property
    def name(self) -> str:
        return self._name
