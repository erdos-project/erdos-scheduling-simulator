import logging
import random
import sys
import uuid
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import absl

from utils import EventTime, fuzz_time, setup_logging

from .graph import Graph
from .resources import Resources
from .tasks import Task, TaskGraph


class FakeRandomNumberGenerator:
    """Alternatively chooses the next condition to execute from the branch.

    Note that the state maintained is per-instance, and if the same instance is passed
    to different branches of the same JobGraph, then the conditions chosen may not be
    as expected.
    """

    def __init__(self):
        self._counter = 0

    def choices(self, population, weights, k):
        """Chooses the next `k` instances in the population, ignoring the prescribed
        weights.

        Args:
            population: The list to make choices from.
            weights: (ignored) The weights to assign to each element in the list.
            k: The number of samples to return.

        Returns:
            A List of `k` samples chosen from the population in order.
        """
        choice_samples = []
        for _ in range(k):
            if self._counter >= len(population):
                self._counter = self._counter % len(population)

            choice_samples.append(population[self._counter])
            self._counter += 1
        return choice_samples


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
        probability: float = 1.0,
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
        self._probability = probability
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
    def probability(self):
        return self._probability

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

        PERIODIC = 1  # Releases a TaskGraph after the period time has elapsed.
        FIXED = 2  # Releases a fixed number of TaskGraphs seperated by the period.

    class ReleasePolicy(object):
        def __init__(
            self,
            policy_type: "ReleasePolicyType",  # noqa: F821
            period: EventTime = EventTime(100, EventTime.Unit.US),
            fixed_invocation_nums: int = 0,
        ) -> None:
            self._policy_type = policy_type
            self._period = period
            self._fixed_invocation_nums = fixed_invocation_nums

        @property
        def policy_type(self) -> "ReleasePolicyType":  # noqa: F821
            return self._policy_type

        @property
        def period(self) -> EventTime:
            return self._period

        @property
        def fixed_invocation_nums(self) -> EventTime:
            return self._fixed_invocation_nums

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
        # Set the seed of the random number generator if provided.
        if _flags:
            # random_number_generator = random.Random(_flags.random_seed)
            random_number_generator = FakeRandomNumberGenerator()
        else:
            random_number_generator = None

        releases = []
        if self.release_policy.policy_type == self.ReleasePolicyType.PERIODIC:
            current_release = start_time
            while current_release <= completion_time:
                releases.append(current_release)
                current_release += self.release_policy.period
        elif self.release_policy.policy_type == self.ReleasePolicyType.FIXED:
            current_release = start_time
            for _ in range(self.release_policy.fixed_invocation_nums):
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
                random_number_generator=random_number_generator,
                _flags=_flags,
            )
        return task_graphs

    def _generate_task_graph(
        self,
        release_time: EventTime,
        task_graph_name: str,
        timestamp: int,
        task_logger: logging.Logger,
        random_number_generator=None,
        _flags: Optional["absl.flags"] = None,
    ) -> TaskGraph:
        """Generates a TaskGraph from the structure of the `JobGraph` whose
        source operators are released at the provided `release_time`.

        Args:
            release_time: The time at which the source operators for the
                `TaskGraph` must be released.
            task_graph_name: The name of the `TaskGraph` being generated.
            timestamp: The timestamp of the `TaskGraph` being generated.
            random_number_generator: The RNG to use for resolving the
                probabilities of execution at Job submission time.
                If `None`, the initial probabilities are forwarded.

        Returns:
            A `TaskGraph` whose source operators are released at the given time.
        """
        # Retrieve variances from the command line flags.
        if _flags:
            runtime_variance = (0, _flags.runtime_variance)
            deadline_variance = (
                _flags.min_deadline_variance,
                _flags.max_deadline_variance,
            )
            use_branch_predicated_deadlines = _flags.use_branch_predicated_deadlines
        else:
            runtime_variance = (0, 0)
            deadline_variance = (0, 0)
            use_branch_predicated_deadlines = False

        # Generate all the `Task`s from the `Job`s in the graph.
        job_to_task_mapping = {}
        for job in self.breadth_first():
            task_release_time = (
                release_time
                if self.is_source(job)
                else EventTime(-1, EventTime.Unit.US)
            )
            task_runtime = fuzz_time(job.runtime, runtime_variance)
            task_deadline = release_time + fuzz_time(
                self.__get_completion_time(),
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
            parent_task = job_to_task_mapping[parent.name]
            task_children = [job_to_task_mapping[child.name] for child in children]
            if parent_task.conditional and random_number_generator is not None:
                # Resolve the conditionals at Job submission time according to
                # the probabilities.
                child_to_release = random_number_generator.choices(
                    population=task_children,
                    weights=[child.probability for child in task_children],
                    k=1,
                )[0]
                for child in task_children:
                    if child == child_to_release:
                        child.update_probability(1.0)
                    else:
                        child.update_probability(0.0)
                        for grand_child_job in self.breadth_first(child.job):
                            if grand_child_job.terminal:
                                break
                            grand_child_task = job_to_task_mapping[grand_child_job.name]
                            grand_child_task.update_probability(0.0)
            task_graph_mapping[parent_task] = task_children

        # Now that the task probabilities have been set, update the deadlines.
        task_graph = TaskGraph(tasks=task_graph_mapping)

        # Compute the deadline based on if the knowledge of branches is assumed or not.
        if use_branch_predicated_deadlines:
            weighted_task_graph_length = sum(
                (
                    task.job.runtime
                    for task in task_graph.get_longest_path(
                        weights=lambda task: task.job.runtime.time
                        if task.probability > sys.float_info.epsilon
                        else 0
                    )
                ),
                start=EventTime.zero(),
            )
        else:
            weighted_task_graph_length = self.__get_completion_time()

        task_graph_deadline = release_time + fuzz_time(
            weighted_task_graph_length, deadline_variance
        )
        for task in task_graph.get_nodes():
            task.update_deadline(task_graph_deadline)

        return task_graph

    def __get_completion_time(self, start=EventTime(0, EventTime.Unit.US)) -> EventTime:
        return sum(
            (
                job.runtime
                for job in self.get_longest_path(
                    weights=lambda job: job.runtime.time
                    if job.probability > sys.float_info.epsilon
                    else 0
                )
            ),
            start=start,
        )

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
