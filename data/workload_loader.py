import json
import logging
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401

from utils import EventTime, setup_logging
from workload import Job, JobGraph, Resource, Resources, Workload


class WorkloadLoader(object):
    """Loads a set of applications from a provided JSON.

    Args:
        json_path (`str`): The path to the JSON file representing the applications.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self, json_path: str, _flags: Optional["absl.flags"] = None) -> None:
        # Set up the logger.
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._resource_logger = setup_logging(
                name="Resources",
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._resource_logger = setup_logging(name="Resources")

        # Read the JSON file for applications and create a JobGraph for
        # each application.
        with open(json_path, "r") as f:
            workload_data = json.load(f)

        if len(workload_data) == 0:
            raise ValueError("Empty Workload generated.")

        # Create the sequence of JobGraphs for each application.
        job_graph_mapping = {}
        for jobs in workload_data:
            job_graph_mapping[jobs["name"]] = WorkloadLoader.load_job_graph(
                jobs["graph"], self._resource_logger
            )

        self._workload = Workload.from_job_graphs(job_graph_mapping)

    @staticmethod
    def load_job_graph(json_repr, resource_logger) -> JobGraph:
        """Load a particular JobGraph from its JSON representation.

        Args:
            json_repr: The JSON representation of the JobGraph.
            resource_logger: The logger to use for Resources.

        Returns:
            A `JobGraph` encoding the serialized JSON representation of the JobGraph.
        """
        job_graph = JobGraph()
        name_to_job_mapping = {}

        # Add all the nodes first to ensure that we can check if the connections
        # were made correctly.
        for node in json_repr:
            # Read characteristics of the Job from the JSON.
            conditional_job = False
            if "conditional" in node and node["conditional"]:
                conditional_job = True
            terminal_job = False
            if "terminal" in node and node["terminal"]:
                terminal_job = True
            runtime = EventTime.zero()
            if "runtime" in node:
                runtime = EventTime(node["runtime"], EventTime.Unit.US)
            resource_requirements = [Resources()]
            if "resource_requirements" in node:
                resource_requirements = WorkloadLoader.__create_resources(
                    resource_requirements=node["resource_requirements"],
                    resource_logger=resource_logger,
                )

            # Create and save the Job.
            job = Job(
                name=node["name"],
                runtime=runtime,
                resource_requirements=resource_requirements,
                conditional=conditional_job,
                terminal=terminal_job,
            )
            name_to_job_mapping[node["name"]] = job
            job_graph.add_job(job=job)

        # Make connections between the nodes.
        for node in json_repr:
            node_job = name_to_job_mapping[node["name"]]
            for child in node["children"]:
                if child not in name_to_job_mapping:
                    raise ValueError(
                        f"Child {child} of {node['name']} was not present in the graph."
                    )
                child_node_job = name_to_job_mapping[child]
                job_graph.add_child(node_job, child_node_job)

        return job_graph

    @staticmethod
    def __create_resources(
        resource_requirements: Sequence[Mapping[str, str]],
        resource_logger: Optional[logging.Logger] = None,
    ) -> Sequence[Resources]:
        """Retrieve the Resource requirements from the given JSON entries.

        Args:
            resource_requirements (`Sequence[Mapping[str, str]]`): The JSON
                entries for the resource requirements of the task.
            logger (`Optional[logging.Logger]`): The logger to use to log the
                results of the execution.

        Returns:
            A Sequence of Resources depicting the potential requirements of a task.
        """
        potential_requirements = []
        for requirements in resource_requirements:
            resource_vector = {}
            for resource, quantity in requirements.items():
                resource_name, resource_id = resource.split(":")
                resource_vector[
                    Resource(name=resource_name, _id=resource_id)
                ] = quantity
            potential_requirements.append(
                Resources(resource_vector=resource_vector, _logger=resource_logger)
            )
        return potential_requirements

    @property
    def workload(self) -> Workload:
        return self._workload
