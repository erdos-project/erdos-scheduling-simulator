import json
import pathlib
import sys
from copy import deepcopy
from typing import Mapping, Optional

import absl  # noqa: F401
import yaml

from data.workload_loader import WorkloadLoader
from utils import EventTime, setup_logging
from workload import JobGraph, Workload, WorkProfile


class WorkloadLoaderClockwork(WorkloadLoader):
    """Loads a set of Clockwork applications from a provided JSON or YAML file.

    Args:
        filename (`str`): The path to the file representing the applications.
        num_instances (`int`): The number of instances to create for each provided
            `WorkProfile` and `JobGraph`.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    NAME_TEMPLATE = "{name}_Instance_{instance}"

    def __init__(
        self,
        filename: str,
        num_instances: int = 15,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        # Most of this is copied and amended from `WorkloadLoader`
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
            self._arrival_period = (
                _flags.override_arrival_period
                if _flags.override_arrival_period > 0
                else None
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._resource_logger = setup_logging(name="Resources")
            self._poisson_arrival_rate = None
            self._arrival_period = None

        # Read the file for applications and create a JobGraph for each application.
        extension = pathlib.Path(filename).suffix.lower()
        with open(filename, "r") as f:
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
        for base_profile in workload_data["profiles"]:
            for instance in range(num_instances):
                profile = deepcopy(base_profile)
                profile["name"] = WorkloadLoaderClockwork.NAME_TEMPLATE.format(
                    name=profile["name"], instance=instance
                )
                work_profile = WorkloadLoader._WorkloadLoader__create_work_profile(
                    profile, self._resource_logger
                )
                work_profiles[work_profile.name] = work_profile

        # Create the sequence of JobGraphs for each application.
        job_graph_mapping = {}
        for index, base_job in enumerate(workload_data["graphs"]):
            for instance in range(num_instances):
                job = deepcopy(base_job)
                # Retrieve the name of the JobGraph.
                if "name" not in job:
                    raise ValueError(
                        f"A name was not defined for the JobGraph {index}."
                    )
                # Amend the job.
                job_name = WorkloadLoaderClockwork.NAME_TEMPLATE.format(
                    name=job["name"], instance=instance
                )
                job["name"] = job_name
                for node in job["graph"]:
                    node["name"] = WorkloadLoaderClockwork.NAME_TEMPLATE.format(
                        name=node["name"], instance=instance
                    )
                    if "work_profile" in node:
                        node[
                            "work_profile"
                        ] = WorkloadLoaderClockwork.NAME_TEMPLATE.format(
                            name=node["work_profile"], instance=instance
                        )

                for required_key in ["graph", "release_policy"]:
                    if required_key not in job:
                        raise ValueError(
                            f'The value for a required key ("{required_key}") '
                            f'was not found in definition of JobGraph ("{job_name}").'
                        )

                # Retrieve the start time from the definition, if provided.
                if "start" in job:
                    start_time = EventTime(job["start"], EventTime.Unit.US)
                else:
                    start_time = EventTime.zero()

                # Retrieve the deadline variance from the definition, if provided.
                if "deadline_variance" in job:
                    deadline_variance = tuple(job["deadline_variance"])
                else:
                    deadline_variance = (0, 0)

                # Construct the ReleasePolicy for the JobGraph.
                if job["release_policy"] == "periodic":
                    if "period" not in job:
                        raise ValueError(
                            f'A "periodic" release policy was requested, but a '
                            f'`period` was not defined for the JobGraph ("{job_name}").'
                        )
                    release_policy = JobGraph.ReleasePolicy.periodic(
                        period=EventTime(
                            job["period"]
                            if self._arrival_period is None
                            else self._arrival_period,
                            EventTime.Unit.US,
                        ),
                        start=start_time,
                    )
                elif job["release_policy"] == "fixed":
                    if "period" not in job or "invocations" not in job:
                        raise ValueError(
                            f'A "fixed" release policy was requested, but either a '
                            f"`period` or `invocations` was not defined for the "
                            f'JobGraph ("{job_name}").'
                        )
                    release_policy = JobGraph.ReleasePolicy.fixed(
                        period=EventTime(
                            job["period"]
                            if self._arrival_period is None
                            else self._arrival_period,
                            EventTime.Unit.US,
                        ),
                        num_invocations=job["invocations"],
                        start=start_time,
                    )
                elif job["release_policy"] == "poisson":
                    if (
                        "rate" not in job and self._poisson_arrival_rate is None
                    ) or "invocations" not in job:
                        raise ValueError(
                            f'A "poisson" release policy was requested, but either a '
                            f"`rate` or `invocations` was not defined for the "
                            f'JobGraph ("{job_name}").'
                        )
                    release_policy = JobGraph.ReleasePolicy.poisson(
                        rate=(
                            job["rate"]
                            if self._poisson_arrival_rate is None
                            else self._poisson_arrival_rate
                        )
                        / num_instances,
                        num_invocations=job["invocations"],
                        start=start_time,
                    )
                else:
                    raise NotImplementedError(
                        f"The release policy {job['release_policy']} "
                        "is not implemented."
                    )

                # Create the JobGraph.
                job_graph_mapping[job["name"]] = WorkloadLoader.load_job_graph(
                    JobGraph(
                        name=job_name,
                        release_policy=release_policy,
                        deadline_variance=deadline_variance,
                    ),
                    job["graph"],
                    work_profiles,
                    EventTime(_flags.slo, EventTime.Unit.US)
                    if _flags.slo >= 0
                    else None,
                )

        self._workload = Workload.from_job_graphs(job_graph_mapping, _flags=_flags)
