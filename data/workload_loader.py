import json
import logging
import pathlib
import sys
from copy import deepcopy
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401
import yaml

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


class WorkloadLoader(object):
    """Loads a set of applications from a provided JSON or YAML file.

    Args:
        path (`str`): The path to the file representing the applications.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self, path: str, _flags: Optional["absl.flags"] = None) -> None:
        # Set up the logger.
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._resource_logger = setup_logging(
                name="Resources",
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._poisson_arrival_rate = (
                _flags.override_poisson_arrival_rate
                if _flags.override_poisson_arrival_rate > sys.float_info.epsilon
                else None
            )
            self._gamma_coefficient = (
                _flags.override_gamma_coefficient
                if _flags.override_gamma_coefficient > sys.float_info.epsilon
                else None
            )
            self._arrival_period = (
                _flags.override_arrival_period
                if _flags.override_arrival_period > 0
                else None
            )
            self._unique_work_profiles = _flags.unique_work_profiles
            self._replication_factor = _flags.replication_factor
            self._slo = (
                EventTime(_flags.override_slo, EventTime.Unit.US)
                if _flags.override_slo > 0
                else EventTime.invalid()
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._resource_logger = setup_logging(name="Resources")
            self._poisson_arrival_rate = None
            self._gamma_coefficient = None
            self._arrival_period = None
            self._unique_work_profiles = False
            self._replication_factor = 1
            self._slo = EventTime.invalid()

        # Read the file for applications and create a JobGraph for each application.
        extension = pathlib.Path(path).suffix.lower()
        with open(path, "r") as f:
            if extension == ".json":
                workload_data = json.load(f)
            elif extension == ".yaml" or extension == ".yml":
                workload_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported extension: {extension}")

        if len(workload_data) == 0:
            raise ValueError("Empty Workload generated.")

        # Create the WorkProfiles provided in the JSON.
        work_profiles: Mapping[str, WorkProfile] = {}
        for profile in workload_data["profiles"]:
            work_profile = WorkloadLoader.__create_work_profile(
                profile, self._resource_logger
            )
            work_profiles[work_profile.name] = work_profile

        # Create the sequence of JobGraphs for each application.
        job_graph_mapping = {}
        for index, job in enumerate(workload_data["graphs"]):
            # Retrieve the name of the JobGraph.
            if "name" not in job:
                raise ValueError(f"A name was not defined for the JobGraph {index}.")
            job_name = job["name"]

            for required_key in ["graph", "release_policy"]:
                if required_key not in job:
                    raise ValueError(
                        f'The value for a required key ("{required_key}") '
                        f'was not found in definition of JobGraph ("{job_name}").'
                    )

            # Retrieve the release policy from the definition.
            release_policy = WorkloadLoader.__create_release_policy(
                job,
                self._arrival_period,
                self._poisson_arrival_rate,
                self._gamma_coefficient,
            )

            # Retrieve the deadline variance from the definition, if provided.
            if "deadline_variance" in job:
                deadline_variance = tuple(job["deadline_variance"])
            else:
                deadline_variance = (0, 0)

            # Create the JobGraph.
            if self._replication_factor > 1:
                for i in range(1, self._replication_factor + 1):
                    job_graph_mapping[
                        f"{job_name}_{i}"
                    ] = WorkloadLoader.load_job_graph(
                        JobGraph(
                            name=f"{job_name}_{i}",
                            release_policy=release_policy,
                            deadline_variance=deadline_variance,
                        ),
                        job["graph"],
                        deepcopy(work_profiles)
                        if not self._unique_work_profiles
                        else work_profiles,
                        self._slo,
                    )
                pass
            else:
                job_graph_mapping[job["name"]] = WorkloadLoader.load_job_graph(
                    JobGraph(
                        name=job_name,
                        release_policy=release_policy,
                        deadline_variance=deadline_variance,
                    ),
                    job["graph"],
                    deepcopy(work_profiles)
                    if not self._unique_work_profiles
                    else work_profiles,
                    self._slo,
                )

        self._workload = Workload.from_job_graphs(job_graph_mapping, _flags=_flags)

    @staticmethod
    def load_job_graph(
        job_graph, json_repr, work_profiles: Mapping[str, WorkProfile], slo: EventTime
    ) -> JobGraph:
        """Load a particular JobGraph from its JSON representation.

        Args:
            job_graph: The `JobGraph` to populate from JSON.
            json_repr: The JSON representation of the JobGraph.
            work_profiles: The `WorkProfile`s loaded from the JSON.
            slo: Overrides each job's SLO with this SLO.

        Returns:
            A `JobGraph` encoding the serialized JSON representation of the JobGraph.
        """
        name_to_job_mapping = {}

        # Add all the nodes first to ensure that we can check if the connections
        # were made correctly.
        for node in json_repr:
            # Read characteristics of the Job from the JSON.
            conditional_job = False
            if "conditional" in node and node["conditional"]:
                conditional_job = True
            probability = 1.0
            if "probability" in node:
                probability = node["probability"]
            terminal_job = False
            if "terminal" in node and node["terminal"]:
                terminal_job = True
            work_profile = None
            if "work_profile" in node:
                work_profile = work_profiles[node["work_profile"]]
            if slo == EventTime.invalid() and "slo" in node:
                slo = EventTime(node["slo"], EventTime.Unit.US)

            # Create and save the Job.
            job = Job(
                name=node["name"],
                profile=work_profile,
                slo=slo,
                conditional=conditional_job,
                probability=probability,
                terminal=terminal_job,
            )
            name_to_job_mapping[node["name"]] = job
            job_graph.add_job(job=job)

        # Make connections between the nodes.
        for node in json_repr:
            node_job = name_to_job_mapping[node["name"]]
            if "children" in node:
                for child in node["children"]:
                    if child not in name_to_job_mapping:
                        raise ValueError(
                            f"Child {child} of {node['name']} was "
                            f"not present in the graph."
                        )
                    child_node_job = name_to_job_mapping[child]
                    job_graph.add_child(node_job, child_node_job)

        return job_graph

    @staticmethod
    def __create_release_policy(
        job: Mapping[str, str],
        override_arrival_period: Optional[int] = None,
        override_poisson_arrival_rate: Optional[float] = None,
        override_gamma_coefficient: Optional[float] = None,
    ) -> JobGraph.ReleasePolicy:
        # Retrieve the start time from the definition, if provided.
        if "start" in job:
            start_time = EventTime(job["start"], EventTime.Unit.US)
        else:
            start_time = EventTime.zero()

        if job["release_policy"] == "periodic":
            if "period" not in job and override_arrival_period is None:
                raise ValueError(
                    'A "periodic" release policy was requested, but a '
                    "`period` was not defined for the JobGraph."
                )
            return JobGraph.ReleasePolicy.periodic(
                period=EventTime(
                    job["period"]
                    if override_arrival_period is None
                    else override_arrival_period,
                    EventTime.Unit.US,
                ),
                start=start_time,
            )
        elif job["release_policy"] == "fixed":
            if (
                "period" not in job or "invocations" not in job
            ) and override_arrival_period is None:
                raise ValueError(
                    'A "fixed" release policy was requested, but either a '
                    "`period` or `invocations` was not defined for the JobGraph."
                )
            return JobGraph.ReleasePolicy.fixed(
                period=EventTime(
                    job["period"]
                    if override_arrival_period is None
                    else override_arrival_period,
                    EventTime.Unit.US,
                ),
                num_invocations=job["invocations"],
                start=start_time,
            )
        elif job["release_policy"] == "poisson":
            if (
                "rate" not in job and override_poisson_arrival_rate is None
            ) or "invocations" not in job:
                raise ValueError(
                    'A "poisson" release policy was requested, but either a '
                    "`rate` or `invocations` was not defined for the JobGraph."
                )
            return JobGraph.ReleasePolicy.poisson(
                rate=job["rate"]
                if override_poisson_arrival_rate is None
                else override_poisson_arrival_rate,
                num_invocations=job["invocations"],
                start=start_time,
            )
        elif job["release_policy"] == "gamma":
            if ("rate" not in job and override_poisson_arrival_rate is None) or (
                "coefficient" not in job and override_gamma_coefficient is None
            ):
                raise ValueError(
                    'A "gamma" release policy was requested, but either a '
                    "`rate` or `coefficient` was not defined for the JobGraph."
                )
            return JobGraph.ReleasePolicy.gamma(
                rate=job["rate"]
                if override_poisson_arrival_rate is None
                else override_poisson_arrival_rate,
                num_invocations=job["invocations"],
                coefficient=job["coefficient"]
                if override_gamma_coefficient is None
                else override_gamma_coefficient,
                start=start_time,
            )
        else:
            raise NotImplementedError(
                f"The release policy {job['release_policy']} is not implemented."
            )

    @staticmethod
    def __create_work_profile(
        profile: Mapping[str, str], resource_logger: Optional[logging.Logger] = None
    ) -> WorkProfile:
        """Retrieves the WorkProfile from the given JSON entries.

        Args:
            profile (`Mapping[str, str]`): The JSON entries for the Work Profile.
            resource_logger (`Optional[logging.Logger]`): The logger instance to use
                to log the results of the execution.

        Returns:
            A `WorkProfile` object that contains the strategies available for execution.
        """
        if "name" not in profile:
            raise KeyError("The key 'name' was not found in the WorkProfile.")

        loading_strategies = ExecutionStrategies()
        if "loading_strategies" in profile:
            loading_strategies = WorkloadLoader.__create_execution_strategies(
                profile["loading_strategies"], resource_logger
            )

        execution_strategies = ExecutionStrategies()
        if "execution_strategies" in profile:
            execution_strategies = WorkloadLoader.__create_execution_strategies(
                profile["execution_strategies"], resource_logger
            )

        return WorkProfile(
            name=profile["name"],
            loading_strategies=loading_strategies,
            execution_strategies=execution_strategies,
        )

    @staticmethod
    def __create_execution_strategies(
        execution_strategies: Sequence[Mapping[str, str]],
        resource_logger: Optional[logging.Logger] = None,
    ) -> ExecutionStrategies:
        """Retrieve the ExecutionStrategies from the given JSON entries.

        Args:
            execution_strategies (`Sequence[Mapping[str, str]]`): The JSON entries for
                the execution strategies for a given Job.
            resource_logger (`Optional[logging.Logger]`): The logger instance to use
                to log the results of the execution.

        Returns:
            An `ExecutionStrategies` object that contains the strategies available for
            execution of the Task.
        """
        execution_strategies_obj = ExecutionStrategies()
        for strategy in execution_strategies:
            resource_requirements = None
            if "resource_requirements" in strategy:
                resource_requirements = WorkloadLoader.__create_resources(
                    strategy["resource_requirements"], resource_logger
                )

            batch_size = 1
            if "batch_size" in strategy:
                batch_size = strategy["batch_size"]

            runtime = EventTime.zero()
            if "runtime" in strategy:
                runtime = EventTime(strategy["runtime"], EventTime.Unit.US)

            execution_strategies_obj.add_strategy(
                ExecutionStrategy(
                    resources=resource_requirements,
                    batch_size=batch_size,
                    runtime=runtime,
                )
            )
        return execution_strategies_obj

    @staticmethod
    def __create_resources(
        resource_requirements: Mapping[str, str],
        resource_logger: Optional[logging.Logger] = None,
    ) -> Resources:
        """Retrieve the Resource requirements from the given JSON entries.

        Args:
            resource_requirements (`Sequence[Mapping[str, str]]`): The JSON
                entries for the resource requirements of the task.
            logger (`Optional[logging.Logger]`): The logger to use to log the
                results of the execution.

        Returns:
            A Resources object depicting the potential requirements of a task.
        """
        resource_vector = {}
        for resource, quantity in resource_requirements.items():
            resource_name, resource_id = resource.split(":")
            resource_vector[Resource(name=resource_name, _id=resource_id)] = quantity
        return Resources(resource_vector=resource_vector, _logger=resource_logger)

    @property
    def workload(self) -> Workload:
        return self._workload
