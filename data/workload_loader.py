import json
from typing import Optional

import absl  # noqa: F401

from utils import EventTime, setup_logging
from workload import JobGraph
from workload.jobs import Job


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
        else:
            self._logger = setup_logging(name=self.__class__.__name__)

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
                jobs["graph"]
            )
        print(job_graph_mapping)

    @staticmethod
    def load_job_graph(json_repr) -> JobGraph:
        """Load a particular JobGraph from its JSON representation.

        Args:
            json_repr: The JSON representation of the JobGraph.

        Returns:
            A `JobGraph` encoding the serialized JSON representation of the JobGraph.
        """
        job_graph = JobGraph()
        name_to_job_mapping = {}

        # Add all the nodes first to ensure that we can check if the connections
        # were made correctly.
        for node in json_repr:
            conditional_job = False
            if "conditional" in node and node["conditional"] == True:
                conditional_job = True
            terminal_job = False
            if "terminal" in node and node["terminal"] == True:
                terminal_job = True
            job = Job(
                name=node["name"],
                runtime=EventTime(node["runtime"], EventTime.Unit.US),
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
