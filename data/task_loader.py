from collections import defaultdict
from operator import attrgetter
from typing import Sequence

from workload import Job, JobGraph, Task, TaskGraph


class TaskLoader(object):
    """Base class for task data loaders."""

    @staticmethod
    def __create_task_graph(tasks: Sequence[Task], job_graph: JobGraph)\
            -> TaskGraph:
        """Constructs the `TaskGraph` from the given set of tasks and their
        job order defined by the `job_graph`.

        This method currently orders the different set of tasks according to
        the topology of the `JobGraph`, and orders the tasks from the same
        `Job` according to the timestamp order.

        Args:
            tasks (`Sequence[Task]`): The set of `Task`s.
            job_graph (`JobGraph`): The topology of the graph.

        Returns:
            A `TaskGraph` defining the order of the different tasks in the
            system.
        """
        # Group all the Tasks by their Job name and order them by timestamp.
        grouped_tasks = defaultdict(list)
        for task in tasks:
            grouped_tasks[task.job.name].append(task)
        for job_name, tasks in grouped_tasks.items():
            grouped_tasks[job_name] = list(
                sorted(tasks, key=attrgetter('timestamp')))

        # Add all the grouped tasks to the TaskGraph with dependencies on the
        # task of the previous timestamp, and parents from the Job with the
        # same timestamp.
        task_graph = defaultdict(list)
        for job_name, tasks in grouped_tasks.items():
            # Add a dependency to the task with the next timestamp.
            for task, child_task in zip(tasks, tasks[1:]):
                task_graph[task].append(child_task)

            # Find all the parent Jobs of each task with the same timestamp and
            # add the task to their children list.
            for task in tasks:
                parent_jobs = job_graph.get_parents(task.job)

                # Find the task with the same timestamp from the Tasks with the
                # name of each job in the parent job.
                for parent_job in parent_jobs:
                    parent_task_same_timestamp = list(
                        filter(
                            lambda parent_task:
                            (parent_task.timestamp == task.timestamp),
                            grouped_tasks[parent_job.name]))

                    # There can be multiple different tasks for the message
                    # and watermark callbacks from the parent Job, and we add
                    # a dependency on all of them here.
                    for parent_task in parent_task_same_timestamp:
                        task_graph[parent_task].append(task)

        return grouped_tasks, TaskGraph(tasks=task_graph)

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded.

        Returns:
            The set of `Job`s loaded.
        """
        raise NotImplementedError(
            "The `get_jobs()` method has not been implemented.")

    def get_job_graph(self) -> JobGraph:
        """Retrieve the constructed `JobGraph`.

        Returns:
            The `JobGraph` constructed by the TaskLoader.
        """
        raise NotImplementedError(
            "The `get_job_graph()` method has not been implemented.")

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoader.

        Returns:
            The set of `Task`s loaded by the TaskLoader.
        """
        raise NotImplementedError(
            "The `get_tasks()` method has not been implemented.")

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoader.

        Returns:
            The `TaskGraph` constructed by the TaskLoader.
        """
        raise NotImplementedError(
            "The `get_task_graph()` method has not been implemented.")
