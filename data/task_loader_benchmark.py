from typing import Optional, Sequence

import absl  # noqa: F401

import utils
from data import TaskLoader
from workload import Job, JobGraph, Resource, Resources, Task, TaskGraph


class TaskLoaderBenchmark(object):
    """Generates a synthetic task workload.

    Args:
        num_jobs (`int`): Number of jobs in the benchmark.
        task_runtime (`int`): Runtime (in us) of benchmark tasks.
        task_deadline (`int`): Relative deadline (in us) of benchmark tasks.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        num_jobs: int = 5,
        task_runtime: int = 15000,
        task_deadline: int = 500000,
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

        self._jobs = [Job(name=f"Job_{index}") for index in range(0, num_jobs)]
        self._job_graph = JobGraph()
        # Note: The jobs are independent.
        for job in self._jobs:
            self._job_graph.add_job(job, [])
        self._tasks = []
        # Creates a single task for each job.
        for index, job in enumerate(self._jobs):
            if index % 2 == 0:
                resource_requirements = Resources(
                    resource_vector={Resource(name="CPU", _id="any"): 1}
                )
            else:
                resource_requirements = Resources(
                    resource_vector={Resource(name="GPU", _id="any"): 1}
                )
            # All times are in us.
            self._tasks.append(
                Task(
                    f"{job.name}_task_{index}",
                    job=job,
                    resource_requirements=resource_requirements,
                    runtime=_flags.task_runtime,
                    deadline=task_deadline,
                    timestamp=0,
                    release_time=0,
                    start_time=-1,
                    completion_time=-1,
                )
            )
        (
            self._grouped_tasks,
            self._task_graph,
        ) = TaskLoader._TaskLoader__create_task_graph(self._tasks, self._job_graph)

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded.

        Returns:
            The set of `Job`s loaded.
        """
        return self._jobs

    def get_job_graph(self) -> JobGraph:
        """Retrieve the constructed `JobGraph`.

        Returns:
            The `JobGraph` constructed by the TaskLoader.
        """
        return self._job_graph

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoader.

        Returns:
            The set of `Task`s loaded by the TaskLoader.
        """
        return self._tasks

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoader.

        Returns:
            The `TaskGraph` constructed by the TaskLoader.
        """
        return self._task_graph
