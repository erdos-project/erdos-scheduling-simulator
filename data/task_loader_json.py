import json
import logging
import sys
from collections import defaultdict
from operator import attrgetter
from random import Random
from typing import Mapping, Optional, Sequence, Tuple

import absl  # noqa: F401
import pydot

import utils
from data import TaskLoader
from workload import Job, JobGraph, Resource, Resources, Task, TaskGraph


class TaskLoaderJSON(object):
    """Loads the Task data from Pylot traces.

    Args:
        graph_path (`str`): The path to the DOT file representing the JobGraph.
        profile_path (`str`): The path to the JSON profile path from Pylot.
        resource_path (`str`): The path to the JSON file of task resource
            requirements.
        max_timestamp (`int`): The maximum timestamp of tasks to load from the
            JSON file.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        graph_path: str,
        profile_path: str,
        resource_path: str,
        max_timestamp: int = sys.maxsize,
        _flags: Optional["absl.flags"] = None,
    ):
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        # Read the data from the profile path.
        with open(profile_path, "r") as f:
            profile_data = json.load(f)
        if len(profile_data) == 0:
            raise ValueError("Empty data returned.")

        # Fix the timestamp offsets from the beginning.
        start_real_time = int(profile_data[0]["ts"])
        start_sim_time = int(profile_data[0]["args"]["timestamp"][1:-1])
        sim_interval = 1
        for entry in profile_data:
            entry_sim_time = int(entry["args"]["timestamp"][1:-1])
            # Figure out the timestamp difference between subsequent timestamps
            # in the profile data, and normalize it to integer timestamps with
            # a difference of 1 starting from 0.
            if sim_interval == 1 and entry_sim_time != start_sim_time:
                sim_interval = entry_sim_time - start_sim_time
            entry["args"]["timestamp"] = int(
                (entry_sim_time - start_sim_time) / sim_interval
            )
            entry["ts"] = entry["ts"] - start_real_time

        # Create the Jobs from the profile path.
        self._jobs = TaskLoaderJSON._TaskLoaderJSON__create_jobs(profile_data)
        self._logger.debug(f"Loaded {len(self._jobs)} Jobs from {profile_path}")

        # Read the DOT file and ensure that we have jobs for all the nodes.
        job_dot_graph = pydot.graph_from_dot_file(graph_path)[0]
        self._logger.debug(f"The DOT graph has {len(job_dot_graph.get_nodes())} Jobs")
        if len(self._jobs) != len(job_dot_graph.get_nodes()):
            raise ValueError(
                f"Mismatch between the Jobs from the DOT graph and the JSON "
                f"profile. JSON profile had {len(self._jobs)} jobs and DOT "
                f"graph had {len(job_dot_graph.get_nodes())} jobs."
            )
        for node in job_dot_graph.get_nodes():
            node_label = node.get_label()
            if node_label not in self._jobs:
                raise ValueError(f"{node_label} found in DOT file, but not in JSON.")

        # Create the JobGraph from the jobs and the DOT representation.
        self._job_graph = TaskLoaderJSON._TaskLoaderJSON__create_job_graph(
            self._jobs,
            map(
                lambda edge: (edge.get_source(), edge.get_destination()),
                job_dot_graph.get_edges(),
            ),
        )

        # Create the Resource requirements from the resource_path.
        resource_logger = utils.setup_logging(
            name="Resources", log_file=_flags.log_file_name, log_level=_flags.log_level
        )
        with open(resource_path, "r") as f:
            self._resource_requirements = (
                TaskLoaderJSON._TaskLoaderJSON__create_resources(
                    json.load(f), resource_logger
                )
            )

        # Create the Tasks and the TaskGraph from the Jobs.
        task_logger = utils.setup_logging(
            name="Task", log_file=_flags.log_file_name, log_level=_flags.log_level
        )
        self._tasks = TaskLoaderJSON._TaskLoaderJSON__create_tasks(
            profile_data,
            self._jobs,
            self._resource_requirements,
            max_timestamp,
            task_logger,
            _flags.deadline_variance,
            Random(42),
        )
        for task in self._tasks:
            self._logger.debug(f"Loaded Task from JSON: {task}")
        self._logger.debug(f"Loaded {len(self._tasks)} Tasks from {profile_path}")
        (
            self._grouped_tasks,
            self._task_graph,
        ) = TaskLoader._TaskLoader__create_task_graph(self._tasks, self._job_graph)
        self._logger.debug("Finished creating TaskGraph from loaded tasks.")

    @staticmethod
    def __create_jobs(json_entries: Sequence[Mapping[str, str]]) -> Mapping[str, Job]:
        """Creates a mapping of Job names to Job instances using the given
        JSON entries.

        Args:
            json_entries (`Sequence[Mapping[str, str]]`): The JSON entries
            retrieved from the profile file.

        Returns:
            A `Mapping[str, Job]` with the Job information retrieved from the
            `json_entries`.
        """
        jobs = {}
        for entry in json_entries:
            if entry["pid"] not in jobs:
                jobs[entry["pid"]] = Job(name=entry["pid"])
        return jobs

    @staticmethod
    def __create_job_graph(
        jobs: Mapping[str, Job], edges: Sequence[Tuple[str, str]]
    ) -> JobGraph:
        """Creates the JobGraph from the given set of Jobs and the relations
        between them as defined by the edges retrieved from the DOT file.

        Args:
            jobs (`Mapping[str, Job]`): The mapping from the Job name to the
                instance of the Job generated for it.
            edges (`Sequence[Tuple[str, str]]`): The relationship between the
                Jobs as defined by the edges of the DOT file.

        Returns:
            A `JobGraph` instance depicting the relation between the different
            `Job`s.
        """
        job_graph = JobGraph()
        for source, destination in edges:
            job_graph.add_job(jobs[source], [jobs[destination]])
        return job_graph

    @staticmethod
    def __create_resources(
        resource_requirements: Sequence[Mapping[str, str]],
        logger: Optional[logging.Logger] = None,
    ) -> Mapping[str, Sequence[Resources]]:
        """Retrieve the Resource requirements from the given JSON entries.

        Args:
            resource_requirements (`Sequence[Mapping[str, str]]`): The JSON
                entries for the resource requirements of each task.
            logger (`Optional[logging.Logger]`): The logger to use to log the
                results of the execution.

        Returns:
            A Mapping of task name (`str`) to a sequence of Resources
            requirements depicting the potential requirements of each task.
        """
        _requirements = {}
        for entry in resource_requirements:
            potential_requirements = []
            for requirements in entry["resource_requirements"]:
                resource_vector = {}
                for resource, quantity in requirements.items():
                    resource_name, resource_id = resource.split(":")
                    resource_vector[
                        Resource(name=resource_name, _id=resource_id)
                    ] = quantity
                potential_requirements.append(
                    Resources(resource_vector=resource_vector, _logger=logger)
                )
            _requirements[entry["name"]] = potential_requirements
        return _requirements

    @staticmethod
    def __create_tasks(
        json_entries: Sequence[Mapping[str, str]],
        jobs: Mapping[str, Job],
        resources: Mapping[str, Sequence[Resources]],
        max_timestamp: int = sys.maxsize,
        logger: Optional[logging.Logger] = None,
        deadline_variance: Optional[int] = 0,
        rng: Random = Random(),
    ) -> Sequence[Task]:
        """Creates a list of tasks from the given JSON entries.

        Args:
            json_entries (`Sequence[Mapping[str, str]]`): The JSON entries
                retrieved from the profile file.
            jobs (`Mapping[str, Job]`): A mapping from the name of the jobs
                to a `Job` instance.
            resources (`Mapping[str, Sequence[Resources]]`): The set of
                potential resources required by each task invocation.
            max_timestamp (`int`): The maximum timestamp of tasks to load from
                the JSON file.
            logger (`Optional[logging.Logger]`): The logger to pass to each
                Task to enable logging of its execution.
            deadline_variance (`Optional[int]`): The % variance to add to
                the assigned deadline for each task.

        Returns:
            A `Sequence[Task]` with the task information retrieved from the
            `json_entries`.
        """
        tasks = []
        for entry in json_entries:
            if entry["args"]["timestamp"] > max_timestamp:
                continue
            # All times are in microseconds.
            runtime_deadline = utils.fuzz_time(rng, entry["dur"], deadline_variance)
            deadline = entry["ts"] + runtime_deadline
            tasks.append(
                Task(
                    name=entry["name"],
                    job=jobs[entry["pid"]],
                    resource_requirements=rng.choice(resources[entry["name"]]),
                    runtime=entry["dur"],
                    deadline=deadline,
                    timestamp=entry["args"]["timestamp"],
                    release_time=entry["ts"],
                    _logger=logger,
                )
            )
        return tasks

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded by the TaskLoaderJSON.

        Returns:
            The set of `Job`s loaded by the TaskLoaderJSON.
        """
        return self._jobs

    def get_job_graph(self) -> JobGraph:
        """Retrieve the `JobGraph` constructed by the TaskLoaderJSON.

        Returns:
            The `JobGraph` constructed by the TaskLoaderJSON.
        """
        return self._job_graph

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoaderJSON.

        Returns:
            The set of `Task`s loaded by the TaskLoaderJSON.
        """
        return self._tasks

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoaderJSON.

        Returns:
            The `TaskGraph` constructed by the TaskLoaderJSON.
        """
        return self._task_graph

    def log_statistics(self):
        """Logs the statistics from the Tasks loaded by the TaskLoaderJSON."""
        for job, tasks in self._grouped_tasks.items():
            # Log the Job name.
            self._logger.debug(f"Job: {job}")

            # Group the tasks by their names.
            tasks_by_task_name = defaultdict(list)
            for task in tasks:
                tasks_by_task_name[task.name].append(task)

            # For each group, log the required statistics.
            for task_name, _tasks in tasks_by_task_name.items():
                # Log the task name.
                self._logger.debug(f"  Task: {task_name}")

                # Log stats about the runtime of the tasks.
                runtimes = list(map(attrgetter("runtime"), _tasks))
                utils.log_statistics(runtimes, self._logger)
