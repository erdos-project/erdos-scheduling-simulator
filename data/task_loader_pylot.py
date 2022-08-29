import json
import logging
import sys
from collections import defaultdict
from typing import Mapping, Optional, Sequence, Tuple

import absl  # noqa: F401
import numpy as np

from data import TaskLoader
from utils import EventTime, fuzz_time, log_statistics, setup_logging
from workload import Job, JobGraph, Resource, Resources, Task, TaskGraph


class TaskLoaderPylot(TaskLoader):
    """Loads the Task data from Pylot traces.

    Args:
        job_graph (`JobGraph`): The JobGraph for the Pylot workload.
        profile_path (`str`): The path to the JSON profile path from Pylot.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        job_graph: JobGraph,
        profile_path: str,
        task_graph_name: str = "pylot_dataflow",
        _flags: Optional["absl.flags"] = None,
    ):
        # Set up the logger.
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)

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
        self._jobs = TaskLoaderPylot._TaskLoaderPylot__create_jobs(profile_data)
        self._logger.debug(f"Loaded {len(self._jobs)} Jobs from {profile_path}")

        # Read the JSON file and ensure that we have jobs for all the nodes.
        self._logger.debug(f"The JobGraph has {len(job_graph)} jobs.")
        if len(self._jobs) != len(job_graph):
            raise ValueError(
                f"Mismatch between the Jobs from the DOT graph and the JSON "
                f"profile. JSON profile had {len(self._jobs)} jobs and DOT "
                f"graph had {len(job_graph)} jobs."
            )

        # Update the Resource requirements from the JobGraph.
        for node in job_graph.get_nodes():
            if node.name not in self._jobs:
                raise ValueError(f"{node.name} found in JobGraph, but not in JSON.")
            job = self._jobs[node.name]
            job._resource_requirements = node.resource_requirements

        # Create the JobGraph from the jobs and the given JobGraph representation.
        self._job_graph = TaskLoaderPylot._TaskLoaderPylot__create_job_graph(
            self._jobs,
            map(
                lambda edge: (edge[0].name, edge[1].name),
                job_graph.get_edges(),
            ),
        )

        # Create the Tasks and the TaskGraph from the Jobs.
        task_logger = setup_logging(
            name="Task", log_file=_flags.log_file_name, log_level=_flags.log_level
        )
        max_timestamp = (
            _flags.max_timestamp if _flags.max_timestamp is not None else sys.max_size
        )
        self._tasks = TaskLoaderPylot._TaskLoaderPylot__create_tasks(
            profile_data,
            task_graph_name,
            self._jobs,
            max_timestamp,
            task_logger,
            (_flags.min_deadline_variance, _flags.max_deadline_variance),
            _flags.synchronize_sensors,
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
        job_to_duration_mapping = defaultdict(list)
        for entry in json_entries:
            job_to_duration_mapping[entry["pid"]].append(entry["dur"])

        jobs = {}
        for job_name, durations in job_to_duration_mapping.items():
            jobs[job_name] = Job(
                name=job_name,
                runtime=EventTime(int(np.mean(durations)), EventTime.Unit.US),
                pipelined=False,
            )

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
        job_graph.pipeline_source_operators()
        return job_graph

    @staticmethod
    def __create_tasks(
        json_entries: Sequence[Mapping[str, str]],
        task_graph_name: str,
        jobs: Mapping[str, Job],
        max_timestamp: int = sys.maxsize,
        logger: Optional[logging.Logger] = None,
        deadline_variance: Optional[Tuple[int, int]] = (0, 0),
        synchronize_sensors: Optional[bool] = False,
    ) -> Sequence[Task]:
        """Creates a list of tasks from the given JSON entries.

        Args:
            json_entries (`Sequence[Mapping[str, str]]`): The JSON entries
                retrieved from the profile file.
            task_graph_name (`str`): The name of the TaskGraph.
            jobs (`Mapping[str, Job]`): A mapping from the name of the jobs
                to a `Job` instance.
            max_timestamp (`int`): The maximum timestamp of tasks to load from
                the JSON file.
            logger (`Optional[logging.Logger]`): The logger to pass to each
                Task to enable logging of its execution.
            deadline_variance (`Optional[Tuple[int, int]]`): The (minimum, maximum) %
                variance to add to the assigned deadline for each task.
            synchronize_sensors (`Optional[bool]`): Ensures that sensors (source jobs)
                release their first task at the same time.

        Returns:
            A `Sequence[Task]` with the task information retrieved from the
            `json_entries`.
        """
        tasks = []
        first_task = defaultdict(lambda: EventTime(0, EventTime.Unit.US))
        for entry in json_entries:
            timestamp = entry["args"]["timestamp"]
            if timestamp > max_timestamp:
                continue
            if timestamp == 0 and synchronize_sensors:
                first_task[entry["name"]] = EventTime(entry["ts"], EventTime.Unit.US)
            offset = first_task[entry["name"]]

            # All times are in microseconds.
            runtime_deadline = fuzz_time(
                EventTime(entry["dur"], EventTime.Unit.US), deadline_variance
            )
            deadline = EventTime(entry["ts"], EventTime.Unit.US) + runtime_deadline
            tasks.append(
                Task(
                    name=entry["name"],
                    task_graph=task_graph_name,
                    job=jobs[entry["pid"]],
                    runtime=EventTime(entry["dur"], EventTime.Unit.US),
                    deadline=(deadline - offset),
                    timestamp=entry["args"]["timestamp"],
                    release_time=EventTime(entry["ts"], EventTime.Unit.US) - offset,
                    _logger=logger,
                )
            )
        return tasks

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded by the TaskLoaderPylot.

        Returns:
            The set of `Job`s loaded by the TaskLoaderPylot.
        """
        return self._jobs

    def get_job_graph(self) -> JobGraph:
        """Retrieve the `JobGraph` constructed by the TaskLoaderPylot.

        Returns:
            The `JobGraph` constructed by the TaskLoaderPylot.
        """
        return self._job_graph

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoaderPylot.

        Returns:
            The set of `Task`s loaded by the TaskLoaderPylot.
        """
        return self._tasks

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoaderPylot.

        Returns:
            The `TaskGraph` constructed by the TaskLoaderPylot.
        """
        return self._task_graph

    def log_statistics(self):
        """Logs the statistics from the Tasks loaded by the TaskLoaderPylot."""
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
                runtimes = list(map(lambda task: task.runtime.time, _tasks))
                log_statistics(runtimes, self._logger)
