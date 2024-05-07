import math
import os
import pathlib
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple

import absl

from utils import EventTime, setup_logging
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    JobGraph,
    Resource,
    Resources,
    Workload,
    WorkProfile,
)

from .base_workload_loader import BaseWorkloadLoader


# Define a Task dataclass for storage of Task information.
@dataclass
class Task:
    name: str
    job: str
    instances: int
    status: str
    start_time: float
    end_time: float
    expected_duration: float
    actual_duration: float
    cpu_requested: float
    cpu_usage: float
    mem_requested: float
    mem_usage: float


class AlibabaTaskUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Task":
            return Task
        return super().find_class(module, name)


FILTERED_DAGS = (
    "j_1442387",
    "j_2583299",
    "j_425976",
    "j_3357093",
    "j_1803334",
    "j_3968360",
    "j_3063832",
    "j_2953716",
)


class AlibabaLoader(BaseWorkloadLoader):
    """Loads the Alibaba trace from the provided file.

    Args:
        path (`str`): The path to a Pickle file containing the Alibaba trace,
            or a folder containing multiple Pickle files.
        workload_interval (`EventTime`): The interval at which to release new
            Workloads.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        workload_interval: EventTime,
        flags: "absl.flags",
    ):
        # Initialize the flags and the logging.
        self._flags = flags
        self._logger = setup_logging(
            name=self.__class__.__name__,
            log_dir=flags.log_dir,
            log_file=flags.log_file_name,
            log_level=flags.log_level,
        )
        self._rng_seed = flags.random_seed
        self._rng = random.Random(self._rng_seed)

        # Initialize data for Workload generation.
        self._workload_paths_and_release_policies = (
            self._construct_workload_definitions()
        )
        self._job_graph_generators: Mapping[str, Callable] = (
            self._initialize_job_graph_generators()
        )
        self._release_times_and_profiles = self._construct_release_times()

        self._job_graphs: Mapping[str, Mapping[str, JobGraph]] = {}
        self._release_times = self._construct_release_times()
        self._current_release_pointer = 0
        self._workload_update_interval = (
            workload_interval
            if not workload_interval.is_invalid()
            else EventTime(sys.maxsize, EventTime.Unit.US)
        )
        self._workload = Workload.empty(flags)
        self._heterogeneous = self._flags.alibaba_enable_heterogeneous_resource_type

        self._task_cpu_multiplier = self._flags.alibaba_loader_task_cpu_multiplier
        self._task_cpu_divisor = self._flags.alibaba_loader_task_cpu_divisor
        self._task_cpu_usage_min = self._flags.alibaba_loader_task_cpu_usage_min
        self._task_cpu_usage_max = self._flags.alibaba_loader_task_cpu_usage_max
        self._task_cpu_usage_random = self._flags.alibaba_loader_task_cpu_usage_random
        self._alibaba_bump_resources_of_low_duration_task = (
            self._flags.alibaba_bump_resources_of_low_duration_task
        )

    def _construct_workload_definitions(
        self,
    ) -> List[Tuple[Optional[str], Optional[JobGraph.ReleasePolicy]]]:
        """Constructs the Workloads that will be used in the simulation.

        This method uses the flags provided to the constructor to determine which
        Workloads to construct.

        Returns:
            A list of tuples of the form:
                (path_to_workload_definition, release_policy)
            If the path_to_workload_definition is None, then the release_policy
            associated with None will be used to construct the Workload.
            Otherwise, the Workload will be constructed from the file at the
            path_to_workload_definition.
        """
        paths_and_policies = []
        if len(self._flags.workload_profile_paths) > 0:
            # If specific files were requested, then we use those.
            if len(self._flags.override_release_policies) != len(
                self._flags.workload_profile_paths
            ):
                raise ValueError(
                    "Number of workload profile paths and release policies must match."
                )
            for index, (path, release_policy_type) in enumerate(
                zip(
                    self._flags.workload_profile_paths,
                    self._flags.override_release_policies,
                )
            ):
                release_policy = None
                if release_policy_type != "poisson":
                    raise NotImplementedError(
                        f"Release policy {release_policy_type} not implemented."
                    )
                else:
                    if index >= len(self._flags.override_poisson_arrival_rates):
                        raise ValueError(
                            "Not enough arrival rates provided for the number of "
                            "workload profile paths."
                        )
                    release_policy = self._construct_release_policy(
                        policy_type=release_policy_type,
                        arrival_rate=float(
                            self._flags.override_poisson_arrival_rates[index]
                        ),
                        num_invocations=int(
                            self._flags.override_num_invocations[index]
                        ),
                    )
                if os.path.isfile(path):
                    extension = pathlib.Path(path).suffix.lower()
                    if extension != ".pkl":
                        raise ValueError(
                            f"Invalid extension {extension} for Alibaba trace."
                        )
                    paths_and_policies.append((path, release_policy))
                else:
                    raise FileNotFoundError(f"No such file or directory: {path}")
        elif self._flags.workload_profile_path is not None:
            path = self._flags.workload_profile_path

            # Construct the release policy.
            release_policy = None
            if self._flags.override_release_policy == "periodic":
                release_policy = self._construct_release_policy(
                    policy_type=self._flags.override_release_policy,
                    arrival_period=EventTime(
                        self._flags.override_arrival_period, EventTime.Unit.US
                    ),
                )
            elif self._flags.override_release_policy == "fixed":
                release_policy = self._construct_release_policy(
                    policy_type=self._flags.override_release_policy,
                    arrival_period=EventTime(
                        self._flags.override_arrival_period, EventTime.Unit.US
                    ),
                    num_invocations=self._flags.override_num_invocation,
                )
            elif self._flags.override_release_policy == "poisson":
                release_policy = self._construct_release_policy(
                    policy_type=self._flags.override_release_policy,
                    arrival_rate=self._flags.override_poisson_arrival_rate,
                    num_invocations=self._flags.override_num_invocation,
                )
            elif self._flags.override_release_policy == "gamma":
                release_policy = self._construct_release_policy(
                    policy_type=self._flags.override_release_policy,
                    arrival_rate=self._flags.override_poisson_arrival_rate,
                    num_invocations=self._flags.override_num_invocation,
                    coefficient=self._flags.override_gamma_coefficient,
                )
            elif self._flags.override_release_policy == "fixed_gamma":
                release_policy = self._construct_release_policy(
                    policy_type=self._flags.override_release_policy,
                    arrival_rate=self._flags.override_poisson_arrival_rate,
                    base_arrival_rate=self._flags.override_base_arrival_rate,
                    num_invocations=self._flags.override_num_invocation,
                    coefficient=self._flags.override_gamma_coefficient,
                )
            else:
                raise NotImplementedError(
                    f"Release policy {self._flags.override_release_policy} not "
                    f"implemented."
                )

            if os.path.isdir(path):
                paths_and_policies.extend(
                    [
                        (os.path.join(path, filename), None)
                        for filename in os.listdir(path)
                        if filename.endswith(".pkl")
                    ]
                )
                paths_and_policies.append((None, release_policy))
            elif os.path.isfile(path):
                extension = pathlib.Path(path).suffix.lower()
                if extension != ".pkl":
                    raise ValueError(
                        f"Invalid extension {extension} for Alibaba trace."
                    )
                paths_and_policies.append((path, release_policy))
            else:
                raise FileNotFoundError(f"No such file or directory: {path}")
        else:
            raise ValueError("No workload profile path provided.")

        if len(paths_and_policies) == 0:
            raise ValueError("No workload profile paths were retrieved.")
        return paths_and_policies

    def _construct_release_policy(
        self,
        policy_type: str,
        arrival_period: Optional[EventTime] = None,
        num_invocations: Optional[int] = None,
        arrival_rate: Optional[float] = None,
        coefficient: Optional[float] = None,
        base_arrival_rate: Optional[float] = None,
    ) -> JobGraph.ReleasePolicy:
        """Constructs the release policy from the given parameters.

        Args:
            policy_type (`str`): The type of release policy to construct.
            arrival_period (`Optional[EventTime]`): The arrival period to use for the
                release policy. Only used for periodic and fixed release policies.
            num_invocations (`Optional[int]`): The number of invocations to use for the
                release policy.
            arrival_rate (`Optional[float]`): The arrival rate to use for the release
                policy.
            coefficient (`Optional[float]`): The coefficient to use for the release
                policy.
            base_arrival_rate (`Optional[float]`): The base arrival rate to use for the
                release policy.

        Returns:
            A release policy object.
        """
        # Construct the start time for this policy.
        start_time = EventTime(
            time=self._rng.randint(
                self._flags.randomize_start_time_min,
                self._flags.randomize_start_time_max,
            ),
            unit=EventTime.Unit.US,
        )
        if policy_type == "periodic":
            if arrival_period is None:
                raise ValueError(
                    "Arrival period must be specified for periodic release policy."
                )
            return JobGraph.ReleasePolicy.periodic(
                period=arrival_period,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif policy_type == "fixed":
            if arrival_period is None:
                raise ValueError(
                    "Arrival period must be specified for fixed release policy."
                )
            if num_invocations is None:
                raise ValueError(
                    "Number of invocations must be specified for fixed release policy."
                )
            return JobGraph.ReleasePolicy.fixed(
                period=arrival_period,
                num_invocations=num_invocations,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif policy_type == "poisson":
            if arrival_rate is None:
                raise ValueError(
                    "Arrival rate must be specified for poisson release policy."
                )
            if num_invocations is None:
                raise ValueError(
                    "Number of invocations must be specified for poisson release "
                    "policy."
                )
            return JobGraph.ReleasePolicy.poisson(
                rate=arrival_rate,
                num_invocations=num_invocations,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif policy_type == "gamma":
            if arrival_rate is None:
                raise ValueError(
                    "Arrival rate must be specified for gamma release policy."
                )
            if num_invocations is None:
                raise ValueError(
                    "Number of invocations must be specified for gamma release policy."
                )
            if coefficient is None:
                raise ValueError(
                    "Coefficient must be specified for gamma release policy."
                )
            return JobGraph.ReleasePolicy.gamma(
                rate=arrival_rate,
                num_invocations=num_invocations,
                coefficient=coefficient,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif policy_type == "fixed_gamma":
            if arrival_rate is None:
                raise ValueError(
                    "Arrival rate must be specified for fixed_gamma release policy."
                )
            if base_arrival_rate is None:
                raise ValueError(
                    "Base arrival rate must be specified for fixed_gamma release "
                    "policy."
                )
            if num_invocations is None:
                raise ValueError(
                    "Number of invocations must be specified for fixed_gamma release "
                    "policy."
                )
            if coefficient is None:
                raise ValueError(
                    "Coefficient must be specified for fixed_gamma release policy."
                )
            return JobGraph.ReleasePolicy.fixed_gamma(
                variable_arrival_rate=arrival_rate,
                base_arrival_rate=base_arrival_rate,
                num_invocations=num_invocations,
                coefficient=coefficient,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        else:
            raise NotImplementedError(
                f"Release policy {self._flags.override_release_policy} not implemented."
            )

    def _construct_release_times(self) -> List[Tuple[EventTime, str]]:
        """Construct the release times of the jobs in the workload.

        Returns:
            A list of release times of the jobs in the workload along with the profile
            from which they are to be chosen.
        """
        # Create the release times.
        release_times = []

        for profile_path, release_policy in self._workload_paths_and_release_policies:
            if release_policy is not None:
                for release_time in release_policy.get_release_times(
                    completion_time=EventTime(
                        self._flags.loop_timeout, EventTime.Unit.US
                    )
                ):
                    release_times.append((release_time, profile_path))

        # Sort the release times.
        release_times.sort()
        return release_times

    def _initialize_job_graph_generators(self) -> Mapping[str, Callable]:
        """Initializes the JobGraph generators for specific Workload profile paths.

        Returns:
            A mapping from the Workload profile path to the JobGraph generator."""

        # Define the JobGraph generator for a given path.
        def job_graph_data_generator(
            path: str,
            min_deadline_variance: int,
            max_deadline_variance: int,
            min_critical_path_runtime: int,
            max_critical_path_runtime: int,
            profile_label: Optional[str] = None,
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"No such file: {path}")
            with open(path, "rb") as pickled_file:
                data: Mapping[str, List[str]] = AlibabaTaskUnpickler(
                    pickled_file
                ).load()
                skipped_job_graphs = 0
                if self._flags.alibaba_dump_filtered_dags:
                    new_pickled_file_name = f"{profile_label}_filtered.pkl"
                    new_pickle = {}
                for job_graph_name, job_tasks in data.items():
                    if job_graph_name in FILTERED_DAGS:
                        continue
                    try:
                        job_graph = self._convert_job_data_to_job_graph(
                            job_graph_name,
                            job_tasks,
                            min_deadline_variance,
                            max_deadline_variance,
                            profile_label,
                        )
                        if job_graph:
                            cp_runtime = job_graph.critical_path_runtime
                            if (
                                min_critical_path_runtime
                                <= cp_runtime.to(EventTime.Unit.US).time
                                < max_critical_path_runtime
                            ):
                                self._job_graphs[path][job_graph_name] = job_graph
                                if self._flags.alibaba_dump_filtered_dags:
                                    new_pickle[job_graph_name] = job_tasks
                            else:
                                skipped_job_graphs += 1
                                # self._logger.debug(
                                #     f"[0] Skipping job graph {job_graph_name} with "
                                #     f"critical path runtime "
                                #     f"{cp_runtime.to(EventTime.Unit.US).time}"
                                #     f" outside of range [{min_critical_path_runtime},"
                                #     f" {max_critical_path_runtime})."
                                # )
                        else:
                            skipped_job_graphs += 1
                    except ValueError as e:
                        self._logger.warning(
                            f"Failed to convert job graph {job_graph_name} "
                            f"with error {e.__class__}: {e}."
                        )
                self._logger.debug(
                    f"[0] Skipped {skipped_job_graphs} job graphs from path {path}, "
                    f"loaded {len(self._job_graphs[path])} job graphs."
                )
                if self._flags.alibaba_dump_filtered_dags:
                    with open(new_pickled_file_name, "wb") as f2:
                        pickle.dump(new_pickle, f2)

        path_to_job_graph_generator_mapping = {}
        for index, (path, _) in enumerate(self._workload_paths_and_release_policies):
            if path is not None:
                min_deadline_variance = (
                    int(self._flags.min_deadline_variance)
                    if index >= len(self._flags.min_deadline_variances)
                    else int(self._flags.min_deadline_variances[index])
                )
                max_deadline_variance = (
                    int(self._flags.max_deadline_variance)
                    if index >= len(self._flags.max_deadline_variances)
                    else int(self._flags.max_deadline_variances[index])
                )
                min_critical_path_runtime = (
                    0
                    if index
                    >= len(self._flags.alibaba_loader_min_critical_path_runtimes)
                    else int(
                        self._flags.alibaba_loader_min_critical_path_runtimes[index]
                    )
                )
                max_critical_path_runtime = (
                    sys.maxsize
                    if index
                    >= len(self._flags.alibaba_loader_max_critical_path_runtimes)
                    else int(
                        self._flags.alibaba_loader_max_critical_path_runtimes[index]
                    )
                )
                profile_label = (
                    None
                    if index >= len(self._flags.workload_profile_path_labels)
                    else self._flags.workload_profile_path_labels[index]
                )
                path_to_job_graph_generator_mapping[path] = partial(
                    job_graph_data_generator,
                    path,
                    min_deadline_variance,
                    max_deadline_variance,
                    min_critical_path_runtime,
                    max_critical_path_runtime,
                    profile_label,
                )
        return path_to_job_graph_generator_mapping

    def _sample_normal_distribution_random(self, n, mean, std, min_val=0, max_val=100):
        samples = []
        while len(samples) < n:
            sample = self._rng.normalvariate(mean, std)
            if min_val <= sample <= max_val:
                samples.append(sample)
        return samples

    def _convert_job_data_to_job_graph(
        self,
        job_graph_name: str,
        job_tasks: List[str],
        min_deadline_variance: Optional[int] = None,
        max_deadline_variance: Optional[int] = None,
        profile_label: Optional[str] = None,
    ) -> Optional[JobGraph]:
        """
        Convert the raw job data to a Job object.

        This method should be implemented according to the specifics of the
        Alibaba trace file format and your Job class.
        """
        # Create the individual Job instances corresponding to each Task.
        task_name_to_simulator_job_mapping = {}
        for task in job_tasks:
            # The name of the Job from the Task.
            job_name = task.name.split("_")[0]

            if self._task_cpu_usage_random:
                # We randomly generate the task CPU utilization between the bounds.
                resource_usage = (
                    self._rng.randint(
                        self._task_cpu_usage_min, self._task_cpu_usage_max
                    )
                    * self._task_cpu_multiplier
                )
                # bump up the resources of low duration tasks
                if (
                    self._alibaba_bump_resources_of_low_duration_task
                    and task.actual_duration < 4
                ):
                    resource_usage = self._task_cpu_usage_max

                job_resources_1 = Resources(
                    resource_vector={
                        Resource(name="Slot_1", _id="any"): resource_usage,
                    }
                )
                job_resources_2 = Resources(
                    resource_vector={
                        Resource(name="Slot_2", _id="any"): resource_usage,
                    }
                )
            else:
                # This code will use the cpu requirements from
                # the alibaba trace and adjust slots
                resource_usage = int(math.ceil(task.cpu_usage / self._task_cpu_divisor))
                job_resources_1 = Resources(
                    resource_vector={
                        # Note: We divide the CPU by some self._task_cpu_divisor instead
                        # of 100 because this would intorduce more variance into the
                        # resource/slots usage.
                        # We used to divide by 100, but the majority of the tasks
                        # would end up using 1 slot, which is not very interesting and
                        # makes no chance for DAG_Sched to do effective packing that
                        # would beat EDF by a significant margin.
                        Resource(name="Slot_1", _id="any"): resource_usage,
                    }
                )
                job_resources_2 = Resources(
                    resource_vector={
                        Resource(name="Slot_2", _id="any"): resource_usage,
                    }
                )
                if resource_usage < self._task_cpu_usage_min:
                    self._logger.debug(
                        "Skipping JobGraph %s because the Job %s required %s units "
                        "of the resource, but the minimum allowed is %s",
                        job_graph_name,
                        job_name,
                        resource_usage,
                        self._task_cpu_usage_min,
                    )
                    return None
                elif resource_usage > self._task_cpu_usage_max:
                    self._logger.debug(
                        "Skipping JobGraph %s because the Job %s required %s units "
                        "of the resource, but the maximum allowed is %s",
                        job_graph_name,
                        job_name,
                        resource_usage,
                        self._task_cpu_usage_max,
                    )
                    return None

            # If we want to try randomizing the duration of the tasks.
            # random_task_duration = round(
            #     self._sample_normal_distribution_random(1, 50, 15)[0]
            # )
            # Use this if we want middle heavy distribution of task durations
            # if i == 0 or i == len(job_tasks) - 1:
            #     random_task_duration =
            #       round(self._sample_normal_distribution_random(1, 10, 5)[0])
            # else:
            #     random_task_duration =
            #       round(self._sample_normal_distribution_random(1, 50, 15)[0])

            if task.actual_duration <= 0:
                # Some loaded TaskGraphs have no duration, skip those.
                self._logger.debug(
                    "Skipping JobGraph %s because the Job %s has duration %s",
                    job_graph_name,
                    job_name,
                    task.actual_duration,
                )
                return None

            job_runtime_1 = EventTime(
                int(math.ceil(task.actual_duration)),
                EventTime.Unit.US,
            )
            # This is used when self._heterogeneous is True
            # to support another execution strategy where it runs faster.
            job_runtime_2 = EventTime(
                int(math.ceil(task.actual_duration * 0.8)),
                EventTime.Unit.US,
            )

            execution_strategies = [
                ExecutionStrategy(
                    resources=job_resources_1,
                    batch_size=1,
                    runtime=job_runtime_1,
                ),
            ]
            if self._heterogeneous:
                execution_strategies.append(
                    ExecutionStrategy(
                        resources=job_resources_2,
                        batch_size=1,
                        runtime=job_runtime_2,
                    ),
                )

            task_name_to_simulator_job_mapping[job_name] = Job(
                name=job_name,
                profile=WorkProfile(
                    name="SlotPolicyFor{}".format(job_name),
                    execution_strategies=ExecutionStrategies(execution_strategies),
                ),
            )

        # Create the JobGraph.
        jobs_to_children = defaultdict(list)
        for task in job_tasks:
            job_and_parents = task.name.split("_", 1)
            if len(job_and_parents) == 1:
                # This job has no parent, add an empty list.
                jobs_to_children[
                    task_name_to_simulator_job_mapping[job_and_parents[0]]
                ].extend([])
            else:
                # This job has children, find them from the list.
                current_job = job_and_parents[0]
                parents = set(job_and_parents[1].split("_"))
                for (
                    parent_job_name,
                    parent_job,
                ) in task_name_to_simulator_job_mapping.items():
                    if parent_job_name[1:] in parents:
                        jobs_to_children[parent_job].append(
                            task_name_to_simulator_job_mapping[current_job]
                        )

        return JobGraph(
            name=(
                job_graph_name
                if profile_label is None
                else f"{job_graph_name}_{profile_label}"
            ),
            jobs=jobs_to_children,
            deadline_variance=(
                (
                    self._flags.min_deadline_variance
                    if min_deadline_variance is None
                    else min_deadline_variance
                ),
                (
                    self._flags.max_deadline_variance
                    if max_deadline_variance is None
                    else max_deadline_variance
                ),
            ),
        )

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        # Get the release times that fit within the range of the current_time and the
        # current_time + workload_interval.
        released_taskgraph_times_and_profiles = []
        while (
            self._current_release_pointer < len(self._release_times)
            and self._release_times[self._current_release_pointer][0]
            <= current_time + self._workload_update_interval
        ):
            released_taskgraph_times_and_profiles.append(
                self._release_times[self._current_release_pointer]
            )
            self._current_release_pointer += 1

        if (
            self._current_release_pointer >= len(self._release_times)
            and len(released_taskgraph_times_and_profiles) == 0
        ):
            # We are at the end of the times, and we didn't release anything this time.
            return None
        else:
            # Choose a random JobGraph and convert it to a TaskGraph to be released.
            task_release_index = 0
            while task_release_index < len(released_taskgraph_times_and_profiles):
                start_time, workload_profile = released_taskgraph_times_and_profiles[
                    task_release_index
                ]
                if workload_profile not in self._job_graphs:
                    self._job_graphs[workload_profile] = {}
                    self._job_graph_generators[workload_profile]()
                job_graph = self._rng.choice(
                    list(self._job_graphs[workload_profile].values())
                )
                task_graph = job_graph.get_next_task_graph(
                    start_time=start_time,
                    _flags=self._flags,
                )
                if (
                    task_graph is not None
                    and task_graph.critical_path_runtime.time < 1000
                    and task_graph.critical_path_runtime.time > 100
                ):
                    self._logger.debug(
                        "[0] Adding TaskGraph %s from path %s to workload with "
                        "release time %s, critical path runtime %s and deadline %s.",
                        task_graph.name,
                        workload_profile,
                        task_graph.release_time,
                        task_graph.critical_path_runtime,
                        task_graph.deadline,
                    )
                    self._workload.add_task_graph(task_graph)
                    task_release_index += 1
            return self._workload
