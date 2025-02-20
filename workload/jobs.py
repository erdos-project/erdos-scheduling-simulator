import math
import random
import sys
import uuid
from enum import Enum
from functools import cached_property
from typing import List, Mapping, Optional, Sequence, Tuple

import absl
import numpy as np

from utils import EventTime, setup_logging

from .graph import Graph
from .profile import WorkProfile
from .strategy import ExecutionStrategies
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
        name (`str`): The name of the ERDOS operator that corresponds to this Job.
        profile (`WorkProfile`): A profile of the computation that the Job is supposed
            to execute (including the execution strategies for different resources).
        slo (`EventTime`): The desired maximum duration of released `Task`s. If
            provided, used to derive the deadline as `deadline = release_time + slo`.
        pipelined (`bool`): True if job's tasks from different timestamps can run
            in parallel.
        conditional (`bool`): True if only some of the job's childrens are invoked
            upon the job's completion instead of all of them.
        probability (`float`): The probability of executing this `Job`, if it's part of
            a conditional graph. 1.0 otherwise.
        terminal (`bool`): True if the job is a terminal job of the conditional
            node (can also be constructed using `create_conditional_pair` method).
    """

    def __init__(
        self,
        name: str,
        profile: Optional[WorkProfile] = None,
        slo: EventTime = EventTime.invalid(),
        pipelined: bool = False,
        conditional: bool = False,
        probability: float = 1.0,
        terminal: bool = False,
    ) -> None:
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._profile = (
            profile
            if profile
            else WorkProfile(
                name=f"{self._name}_{self._id}_work_profile",
                execution_strategies=ExecutionStrategies(),
            )
        )
        self._slo = slo
        self._pipelined = pipelined
        self._conditional = conditional
        self._probability = probability
        self._terminal = terminal

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return str(self._id)

    @property
    def profile(self) -> WorkProfile:
        return self._profile

    @property
    def execution_strategies(self) -> ExecutionStrategies:
        return self._profile.execution_strategies

    @property
    def slo(self) -> EventTime:
        return self._slo

    @property
    def pipelined(self) -> bool:
        return self._pipelined

    @property
    def conditional(self) -> bool:
        return self._conditional

    @property
    def probability(self) -> float:
        return self._probability

    @property
    def terminal(self) -> bool:
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
        name (`str`): The name to assign to this `JobGraph`.
        jobs (`Optional[Mapping[Job, Sequence[Job]]]`): A Mapping from a set of `Job`s
            to their children that needs to be initialized into a `JobGraph`.
        release_policy (`Optional[ReleasePolicy]`): The policy to use to generate
            `TaskGraph`s from this `JobGraph`.
        completion_time: The completion time to assign to this `JobGraph`. If `None`,
            the completion time will be computed as the sum of the runtime of the
            longest path in the graph.
        deadline_variance (`Optional[Tuple[int, int]]`): The variance by which the
            deadline can be dithered on top of the actual execution time.
    """

    RELEASE_POLICIES = (
        "periodic",
        "fixed",
        "poisson",
        "gamma",
        "closed_loop",
        "fixed_gamma",
    )

    class ReleasePolicyType(Enum):
        """Represents the different release policies supported by a JobGraph."""

        # Releases a TaskGraph after the period time has elapsed.
        PERIODIC = 1
        # Releases a fixed number of TaskGraphs seperated by the period.
        FIXED = 2
        # Releases a fixed number of TaskGraphs seperated by a Poisson arrival rate.
        POISSON = 3
        # Releases a fixed number of TaskGraphs seperated by a Gamma arrival rate.
        GAMMA = 4
        # Releases a fixed number of TaskGraphs in a closed loop fashion (i.e., a new
        # job graph is released whenever the previous one completes).
        CLOSED_LOOP = 5
        # Releases a fixed number of TaskGraphs seperated by a Gamma arrival rate with
        # some base arrival rate.
        FIXED_AND_GAMMA = 6

    class ReleasePolicy(object):
        """A class representing the parameters of the release policy by which the
        instances of a `TaskGraph` are generated from this `JobGraph`.

        See `JobGraph.generate_task_graphs` for more details.

        The class instance should be initialized from the specific `staticmethod`s
        corresponding to the `ReleasePolicyType` as opposed to the `__init__` method.
        """

        def __init__(
            self,
            policy_type: "ReleasePolicyType",  # noqa: F821
            period: EventTime,
            fixed_invocation_nums: int,
            variable_arrival_rate: float,
            coefficient: float,
            concurrency: int,
            start: EventTime,
            base_arrival_rate: Optional[
                float
            ] = None,  # Only used in fix_and_gamma policy
            rng_seed: Optional[int] = None,
        ) -> None:
            self._policy_type = policy_type
            self._period = period
            self._fixed_invocation_nums = fixed_invocation_nums
            self._variable_arrival_rate = variable_arrival_rate
            self._base_arrival_rate = base_arrival_rate
            self._coefficient = coefficient
            self._concurrency = concurrency
            self._start = start
            self._rng = (
                np.random.default_rng()
                if rng_seed is None
                else np.random.default_rng(seed=rng_seed)
            )

        def get_release_times(self, completion_time: EventTime) -> List[EventTime]:
            """Calculate the release times for this policy up-till the completion time.

            Args:
                completion_time (`EventTime`): The time upto which the release policy
                    needs to be extrapolated.
            """
            releases = []
            if self._fixed_invocation_nums == 0:
                # If there are no invocations, we return an empty list.
                return releases

            if self._policy_type == JobGraph.ReleasePolicyType.PERIODIC:
                releases.extend(
                    map(
                        lambda time: EventTime(int(time), EventTime.Unit.US),
                        np.arange(
                            self._start.to(EventTime.Unit.US).time,
                            completion_time.to(EventTime.Unit.US).time,
                            self._period.to(EventTime.Unit.US).time,
                        ),
                    )
                )
            elif self._policy_type == JobGraph.ReleasePolicyType.FIXED:
                releases.extend(
                    map(
                        lambda time: EventTime(int(time), EventTime.Unit.US),
                        np.linspace(
                            self._start.to(EventTime.Unit.US).time,
                            self._start.to(EventTime.Unit.US).time
                            + (
                                self._period.to(EventTime.Unit.US).time
                                * self._fixed_invocation_nums
                            ),
                            num=self._fixed_invocation_nums,
                            endpoint=False,
                        ),
                    )
                )
            elif self._policy_type == JobGraph.ReleasePolicyType.POISSON:
                inter_arrival_times = self._rng.poisson(
                    1 / self._variable_arrival_rate, self._fixed_invocation_nums - 1
                )
                current_release = self._start
                releases.append(current_release)
                for inter_arrival_time in inter_arrival_times:
                    current_release += EventTime(
                        int(inter_arrival_time), EventTime.Unit.US
                    )
                    releases.append(current_release)
            elif self._policy_type == JobGraph.ReleasePolicyType.GAMMA:
                # inter_arrival_times = np.clip(
                #     self._rng.gamma(
                #         (1 / self._coefficient),
                #         self._coefficient / self._arrival_rate,
                #         size=self._fixed_invocation_nums - 1,
                #     ),
                #     a_min=2500,  # Maintain a minimum rate of 2500µs between releases.
                #     a_max=None,
                # )
                inter_arrival_times = self._rng.gamma(
                    (1 / self._coefficient),
                    self._coefficient / self._variable_arrival_rate,
                    size=self._fixed_invocation_nums - 1,
                )
                current_release_time = self._start.time
                releases.append(
                    EventTime(round(current_release_time), EventTime.Unit.US)
                )
                for inter_arrival_time in inter_arrival_times:
                    current_release_time += inter_arrival_time
                    releases.append(
                        EventTime(round(current_release_time), EventTime.Unit.US)
                    )
            elif self._policy_type == JobGraph.ReleasePolicyType.CLOSED_LOOP:
                # Release the first set of Tasks at the start time.
                num_releases = (
                    self._concurrency
                    if self._fixed_invocation_nums >= self._concurrency
                    else self._fixed_invocation_nums
                )
                releases.extend([self._start] * num_releases)
            elif self._policy_type == JobGraph.ReleasePolicyType.FIXED_AND_GAMMA:
                # Given base_arrival_rate, variable_arrival_rate, start_time, and
                # fixed_invocation_nums, we want to # solve for end_time
                # (self._base_arrival_rate + self._variable_arrival_rate) *
                # (end_time - start_time) = self._fixed_invocation_nums
                start_time = self._start.to(EventTime.Unit.US).time
                end_time = (
                    int(
                        self._fixed_invocation_nums
                        / (self._base_arrival_rate + self._variable_arrival_rate)
                    )
                    + start_time
                )
                gamma_policy_invocations = math.floor(
                    self._variable_arrival_rate * (end_time - start_time)
                )
                fixed_policy_invocations = math.floor(
                    self._base_arrival_rate * (end_time - start_time)
                )
                # Due to rounding, we may have less invocations than the
                # fixed_invocation_nums. We will add the difference to the
                # gamma_policy_invocations.
                gamma_policy_invocations += self._fixed_invocation_nums - (
                    fixed_policy_invocations + gamma_policy_invocations
                )

                # First we apply gamma release policy
                inter_arrival_times = self._rng.gamma(
                    (1 / self._coefficient),
                    self._coefficient / self._variable_arrival_rate,
                    size=gamma_policy_invocations - 1,
                )
                current_release_time = self._start.time
                releases.append(
                    EventTime(round(current_release_time), EventTime.Unit.US)
                )
                for inter_arrival_time in inter_arrival_times:
                    current_release_time += inter_arrival_time
                    releases.append(
                        EventTime(round(current_release_time), EventTime.Unit.US)
                    )

                # Then we apply fixed release policy
                releases.extend(
                    map(
                        lambda time: EventTime(int(time), EventTime.Unit.US),
                        np.linspace(
                            start_time,
                            end_time,
                            num=fixed_policy_invocations,
                            endpoint=False,
                        ),
                    )
                )
                releases.sort()
            else:
                raise NotImplementedError(
                    f"The policy {self._policy_type} has not been implemented yet."
                )
            return releases

        @staticmethod
        def periodic(
            period: EventTime,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `PERIODIC` release policy.

            Args:
                period (`EventTime`): The exact time between two successive
                    invocations of the `TaskGraph`.
                start (`EventTime`): The time at which the periodic release of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.PERIODIC,
                period=period,
                fixed_invocation_nums=-1,
                variable_arrival_rate=-1.0,
                coefficient=-1.0,
                concurrency=0,
                start=start,
                rng_seed=rng_seed,
            )

        @staticmethod
        def fixed(
            period: EventTime,
            num_invocations: int,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `FIXED` release policy.

            Args:
                period (`EventTime`): The exact time between two successive
                    invocations of the `TaskGraph`.
                num_invocations (`int`): The number of invocations of the `TaskGraph`.
                start (`EventTime`): The time at which the periodic release of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.FIXED,
                period=period,
                fixed_invocation_nums=num_invocations,
                variable_arrival_rate=-1.0,
                coefficient=-1.0,
                concurrency=0,
                start=start,
                rng_seed=rng_seed,
            )

        @staticmethod
        def poisson(
            rate: float,
            num_invocations: int,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `POISSON` release policy.

            Args:
                rate (`float`): The lambda (rate) parameter defining the Poisson
                    arrival distribution.
                num_invocations (`int`): The number of invocations of the `TaskGraph`.
                start (`EventTime`): The time at which the poisson arrival of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.POISSON,
                period=EventTime.invalid(),
                fixed_invocation_nums=num_invocations,
                variable_arrival_rate=rate,
                coefficient=-1.0,
                concurrency=0,
                start=start,
                rng_seed=rng_seed,
            )

        @staticmethod
        def gamma(
            rate: float,
            coefficient: float,
            num_invocations: int,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `GAMMA` release policy.

            Args:
                rate (`float`): The lambda (rate) parameter defining the Gamma
                    arrival distribution.
                coefficient (`float`): The coefficient parameter defining the Gamma
                    arrival distribution.
                num_invocations (`int`): The number of invocations of the `TaskGraph`.
                start (`EventTime`): The time at which the poisson arrival of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.GAMMA,
                period=EventTime.invalid(),
                fixed_invocation_nums=num_invocations,
                variable_arrival_rate=rate,
                coefficient=coefficient,
                concurrency=0,
                start=start,
                rng_seed=rng_seed,
            )

        @staticmethod
        def fixed_gamma(
            variable_arrival_rate: float,
            base_arrival_rate: float,
            coefficient: float,
            num_invocations: int,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `GAMMA` release policy.

            Args:
                variable_arrival_rate (`float`): The lambda (rate) parameter defining
                    the Gamma arrival distribution.
                base_arrival_rate (`float`): The base arrival rate of the fixed release
                    policy.
                coefficient (`float`): The coefficient parameter defining the Gamma
                    arrival distribution.
                num_invocations (`int`): The number of invocations of the `TaskGraph`.
                start (`EventTime`): The time at which the poisson arrival of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.FIXED_AND_GAMMA,
                period=EventTime.invalid(),
                fixed_invocation_nums=num_invocations,
                variable_arrival_rate=variable_arrival_rate,
                base_arrival_rate=base_arrival_rate,
                coefficient=coefficient,
                concurrency=0,
                start=start,
                rng_seed=rng_seed,
            )

        @staticmethod
        def closed_loop(
            concurrency: int,
            num_invocations: int,
            start: EventTime = EventTime.zero(),
            rng_seed: Optional[int] = None,
        ) -> "ReleasePolicy":  # noqa: F821
            """Creates the parameters corresponding to the `CLOSED_LOOP` release policy.

            Args:
                concurrency (`int`): The number of concurrent `TaskGraph`s to execute.
                num_invocations (`int`): The total number of invocations of the
                    `TaskGraph`.
                start (`EventTime`): The time at which the closed loop execution of the
                    `TaskGraph`s should begin.
                rng_seed (`Optional[int]`): The seed to use for the random number
                    generation.

            Returns:
                A `ReleasePolicy` instance with the required parameters.
            """
            if concurrency == 0 or num_invocations == 0:
                raise RuntimeError("The concurrency and num_invocations must be > 0.")
            return JobGraph.ReleasePolicy(
                policy_type=JobGraph.ReleasePolicyType.CLOSED_LOOP,
                period=EventTime.invalid(),
                fixed_invocation_nums=num_invocations,
                variable_arrival_rate=-1.0,
                coefficient=-1.0,
                concurrency=concurrency,
                start=start,
            )

        @property
        def policy_type(self) -> "ReleasePolicyType":  # noqa: F821
            return self._policy_type

        @property
        def period(self) -> EventTime:
            if self.policy_type not in (
                JobGraph.ReleasePolicyType.PERIODIC,
                JobGraph.ReleasePolicyType.FIXED,
            ):
                raise ValueError(
                    "The `period` parameter is only available in `PERIODIC` "
                    "and `FIXED` release policy types."
                )
            return self._period

        @property
        def num_invocations(self) -> int:
            if self.policy_type not in (
                JobGraph.ReleasePolicyType.FIXED,
                JobGraph.ReleasePolicyType.POISSON,
                JobGraph.ReleasePolicyType.GAMMA,
                JobGraph.ReleasePolicyType.CLOSED_LOOP,
            ):
                raise ValueError(
                    "The `num_invocations` parameter is only available in `FIXED` "
                    ",`POISSON`, `GAMMA` and `CLOSED_LOOP` release policy types."
                )
            return self._fixed_invocation_nums

        @property
        def rate(self) -> float:
            if self.policy_type not in (
                JobGraph.ReleasePolicyType.POISSON,
                JobGraph.ReleasePolicyType.GAMMA,
            ):
                raise ValueError(
                    "The `rate` parameter is only available in `POISSON` policy ."
                )
            return self._variable_arrival_rate

        @property
        def coefficient(self) -> float:
            if self.policy_type != JobGraph.ReleasePolicyType.GAMMA:
                raise ValueError(
                    "The `coefficient` parameter is only available in `GAMMA` policy."
                )
            return self._coefficient

        @property
        def concurrency(self) -> int:
            """The number of concurrent requests to send out."""
            if self.policy_type != JobGraph.ReleasePolicyType.CLOSED_LOOP:
                raise ValueError(
                    "The `concurrency` parameter is only available in `CLOSED_LOOP` "
                    "policy."
                )
            return self._concurrency

        @property
        def start_time(self) -> EventTime:
            return self._start

        def __str__(self) -> str:
            return (
                f"ReleasePolicy(policy_type={self.policy_type}, period={self._period}, "
                f"num_invocations={self._fixed_invocation_nums}, "
                f"rate={self._variable_arrival_rate}, coefficient={self._coefficient}, "
                f"concurrency={self._concurrency}, start_time={self.start_time})"
            )

    def __init__(
        self,
        name: str,
        jobs: Optional[Mapping[Job, Sequence[Job]]] = {},
        release_policy: Optional["ReleasePolicy"] = None,
        completion_time: Optional[EventTime] = None,
        deadline_variance: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(jobs)
        self._name = name
        self._release_policy = release_policy
        # If completion_time is None, we'll calculate it at:
        # @property
        # def completion_time(self):
        self._completion_time = completion_time
        self._deadline_variance = deadline_variance
        self._remaining_task_graphs = sys.maxsize
        self._task_graph_index = 0

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

    def get_next_task_graph(
        self,
        start_time: EventTime,
        _flags: Optional["absl.flags"] = None,
    ) -> Optional[TaskGraph]:
        if self._remaining_task_graphs > 0:
            self._remaining_task_graphs -= 1
            self._task_graph_index += 1
            return self._generate_task_graph(
                release_time=start_time,
                task_graph_name=f"{self.name}@{self._task_graph_index}",
                timestamp=self._task_graph_index,
                _flags=_flags,
            )
        else:
            return None

    def generate_task_graphs(
        self,
        completion_time: EventTime,
        _flags: Optional["absl.flags"] = None,
    ) -> Mapping[str, TaskGraph]:
        """Generates the task graphs under the defined `release_policy`
        and the `completion_time`.

        Args:
            completion_time: The time at which the simulator has to end.

        Returns:
            A mapping from the name of the `TaskGraph` to the `TaskGraph`.
        """
        releases = self._release_policy.get_release_times(completion_time)
        if self.release_policy.policy_type == self.ReleasePolicyType.CLOSED_LOOP:
            self._remaining_task_graphs = self.release_policy.num_invocations - len(
                releases
            )

        # Generate the task graphs for all the releases.
        task_graphs = {}
        for index, release_time in enumerate(releases):
            task_graph_name = f"{self.name}@{index}"
            task_graphs[task_graph_name] = self._generate_task_graph(
                release_time=release_time,
                task_graph_name=task_graph_name,
                timestamp=index,
                _flags=_flags,
            )
        self._task_graph_index = len(task_graphs) - 1
        return task_graphs

    def _generate_task_graph(
        self,
        release_time: EventTime,
        task_graph_name: str,
        timestamp: int,
        _flags: Optional["absl.flags"] = None,
    ) -> TaskGraph:
        """Generates a TaskGraph from the structure of the `JobGraph` whose
        source operators are released at the provided `release_time`.

        Args:
            release_time: The time at which the source operators for the
                `TaskGraph` must be released.
            task_graph_name: The name of the `TaskGraph` being generated.
            timestamp: The timestamp of the `TaskGraph` being generated.

        Returns:
            A `TaskGraph` whose source operators are released at the given time.
        """
        # Retrieve required parameters from the command line flags.
        if _flags:
            # random_number_generator = random.Random(_flags.random_seed)
            random_number_generator = FakeRandomNumberGenerator()
            if self._deadline_variance is None:
                deadline_variance = (
                    _flags.min_deadline_variance,
                    _flags.max_deadline_variance,
                )
            else:
                deadline_variance = self._deadline_variance
            deadline_bounds = (_flags.min_deadline, _flags.max_deadline)
            use_branch_predicated_deadlines = _flags.use_branch_predicated_deadlines
            resolve_conditionals = _flags.resolve_conditionals_at_submission
            task_logger = setup_logging(
                name="Task",
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            random_number_generator = None
            deadline_variance = (
                self._deadline_variance
                if self._deadline_variance is not None
                else (0, 0)
            )
            deadline_bounds = (0, sys.maxsize)
            use_branch_predicated_deadlines = False
            resolve_conditionals = False
            task_logger = setup_logging(name="Task")

        # Create an RNG to be used when fuzzing deadlines, seeded by
        # the TaskGraph name and the global random seed, if provided.
        # This ensures that deadlines are deterministic, which is
        # needed for simulator/Spark parity.
        deadline_rng = random.Random(
            (str(_flags.random_seed) if _flags else "") + task_graph_name
        )

        # Generate the deadline for all the Tasks.
        # TODO (Sukrit): Right now, this assumes that all Tasks in the TaskGraph come
        # with the same deadline. At some point, we will have to implement a
        # heuristic-based deadline splitting technique.
        
        # NOTE: The taskgraph deadline is re-generated (and overwritten) after 
        # use_branch_predicated_deadlines code, since fuzz is invoked again there.
        task_deadline = release_time + self.completion_time.fuzz(
            deadline_variance, deadline_bounds, rng=deadline_rng
        )

        # Generate all the `Task`s from the `Job`s in the graph.
        job_to_task_mapping = {}
        for job in self.breadth_first():
            task_release_time = (
                release_time
                if self.is_source(job)
                else EventTime(-1, EventTime.Unit.US)
            )
            job_to_task_mapping[job.name] = Task(
                name=job.name,
                task_graph=task_graph_name,
                job=job,
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
            if (
                resolve_conditionals
                and parent_task.conditional
                and random_number_generator is not None
            ):
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
        task_graph = TaskGraph(
            name=task_graph_name, tasks=task_graph_mapping, job_graph=self
        )

        # Compute the deadline based on if the knowledge of branches is assumed or not.
        if use_branch_predicated_deadlines:
            weighted_task_graph_length = sum(
                (
                    task.job.runtime
                    for task in task_graph.get_longest_path(
                        weights=lambda task: (
                            task.job.runtime.time
                            if task.probability > sys.float_info.epsilon
                            else 0
                        )
                    )
                ),
                start=EventTime.zero(),
            )
        else:
            weighted_task_graph_length = self.__get_completion_time()

        # NOTE: This is the second time the deadline is being set, based on a second 
        # invocation of fuzz.
        task_graph_deadline = release_time + weighted_task_graph_length.fuzz(
            deadline_variance, deadline_bounds, rng=deadline_rng
        )
        if _flags and _flags.decompose_deadlines:
            stages_info = {}
            stages = set([])
            for task in task_graph.topological_sort():
                stage = 0
                for previous_task in task_graph.get_parents(task):
                    stage = max(stage, stages_info.get(previous_task, 0) + 1)
                stages_info[task] = stage
                stages.add(stage)

            critical_path = task_graph.get_longest_path(
                weights=lambda task: (task.slowest_execution_strategy.runtime.time)
            )
            critical_path_time = (
                sum(
                    [t.slowest_execution_strategy.runtime for t in critical_path],
                    start=EventTime.zero(),
                )
                .to(EventTime.Unit.US)
                .time
            )
            stage_wise_deadline = {}
            for critical_task in critical_path:
                stage_deadline = int(
                    task_graph_deadline.to(EventTime.Unit.US).time
                    * critical_task.slowest_execution_strategy.runtime.to(
                        EventTime.Unit.US
                    ).time
                    / critical_path_time
                )
                stage_wise_deadline[stages_info[critical_task]] = stage_deadline

            for task in task_graph.get_nodes():
                # For the tasks that do not fall on the critical path of the
                # computation, we heuristically find the closest stage and assign it
                # to that stage's deadline.
                task_stage = min(
                    stage_wise_deadline.keys(),
                    key=lambda s: abs(
                        s - stages_info[task]
                        if s >= stages_info[task]
                        else float("inf")
                    ),
                )
                deadline = EventTime(
                    stage_wise_deadline[task_stage], unit=EventTime.Unit.US
                )
                task.update_deadline(deadline)
        else:
            for task in task_graph.get_nodes():
                task.update_deadline(task_graph_deadline)

        return task_graph

    def __get_completion_time(self, start=EventTime.zero()) -> EventTime:
        return sum(
            (
                (
                    job.slo
                    if job.slo != EventTime.invalid()
                    else job.execution_strategies.get_slowest_strategy().runtime
                )
                for job in self.get_longest_path(
                    weights=lambda job: (
                        job.execution_strategies.get_slowest_strategy()
                        .runtime.to(EventTime.Unit.US)
                        .time
                        if job.probability > sys.float_info.epsilon
                        else 0
                    )
                )
            ),
            start=start,
        )

    @property
    def completion_time(self) -> EventTime:
        if not self._completion_time and len(self) != 0:
            self._completion_time = self.__get_completion_time()
        return self._completion_time

    @cached_property
    def critical_path_runtime(self) -> EventTime:
        return sum(
            [
                job.execution_strategies.get_slowest_strategy().runtime
                for job in self.get_longest_path(
                    weights=lambda job: (
                        job.execution_strategies.get_slowest_strategy()
                        .runtime.to(EventTime.Unit.US)
                        .time
                        if job.probability > sys.float_info.epsilon
                        else 0
                    )
                )
            ],
            start=EventTime.zero(),
        )

    @property
    def release_policy(self) -> Optional["ReleasePolicy"]:
        return self._release_policy

    @property
    def name(self) -> str:
        return self._name
