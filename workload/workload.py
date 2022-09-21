from typing import Callable, Mapping, Optional, Sequence

import absl

from utils import EventTime, setup_logging

from .jobs import JobGraph
from .tasks import Task, TaskGraph


class Workload(object):
    """A representation of the Workload that is to be simulated.

    The Workload can be continually generated from the set of JobGraphs provided
    to it, or might have a single TaskGraph provided to it that is released at
    the beginning of the simulation.

    The class instance should be initialized from the `from_job_graphs` or
    `from_task_graphs` method.
    """

    def __init__(
        self,
        job_graphs: Optional[Mapping[str, JobGraph]] = None,
        task_graphs: Optional[Mapping[str, TaskGraph]] = None,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        if job_graphs is None and task_graphs is None:
            raise ValueError(
                "Either the JobGraph(s) or TaskGraph(s) should be provided."
            )

        # Setup the logger and save the flags.
        self._flags = _flags
        if self._flags:
            self._logger = setup_logging(
                "Workload", log_file=_flags.log_file_name, log_level=_flags.log_level
            )
        else:
            self._logger = setup_logging("Workload")

        # Prioritize the creation of the Workload from a TaskGraph
        # for backward compatibility with older versions of logs.
        if task_graphs is None:
            self._job_graphs = job_graphs
            self._task_graphs = dict()
            self._initialized = False
        else:
            self._job_graphs = dict()
            self._task_graphs = task_graphs
            self._initialized = True

    @staticmethod
    def from_job_graphs(
        job_graphs: Mapping[str, JobGraph], _flags: Optional["absl.flags"] = None
    ) -> "Workload":
        """Creates a Workload from the given JobGraphs.

        Args:
            job_graphs: A mapping from the name of the application to its JobGraph.

        """
        return Workload(job_graphs=job_graphs, _flags=_flags)

    @staticmethod
    def from_task_graphs(
        task_graphs: Mapping[str, TaskGraph], _flags: Optional["absl.flags"] = None
    ) -> "Workload":
        """Creates a Workload from the given TaskGraphs.

        Args:
            task_graphs: A mapping from the name of the application to its TaskGraph.
        """
        return Workload(task_graphs=task_graphs, _flags=_flags)

    @staticmethod
    def empty() -> "Workload":
        """Creates an empty Workload.

        Returns:
            An empty `Workload` that has no TaskGraphs.
        """
        return Workload.from_task_graphs(task_graphs={})

    def populate_task_graphs(
        self,
        completion_time: EventTime,
    ) -> None:
        """Populates the instantiations of the `TaskGraph`s according to the
        given `JobGraph`s and the release policy.

        Args:
            completion_time (`EventTime`): The time at which the simulator is
                supposed to end.

        Raises:
            `ValueError` if the sys.maxsize value is passed for the `completion_time`
            and the `Workload` was not instantiated from the `TaskGraph`s.
        """
        if not self._initialized:
            for job_graph in self._job_graphs.values():
                self._task_graphs |= job_graph.generate_task_graphs(
                    completion_time, _flags=self._flags
                )
            self._initialized = True

    def get_job_graph(self, name: str) -> Optional[JobGraph]:
        """Retrieves the JobGraph for the given application, if present.

        Args:
            name (`str`): The name of the application.

        Returns:
            The `JobGraph` representing the application, if available.
        """
        return self._job_graphs.get(name)

    def get_task_graph(self, name: str) -> Optional[TaskGraph]:
        """Retrieves the TaskGraph for the given application, if present.

        Args:
            name (`str`): The name of the application.

        Returns:
            The `TaskGraph` representing the application, if available.
        """
        return self._task_graphs.get(name)

    def notify_task_completion(
        self,
        task: Task,
        finish_time: EventTime,
        csv_logger=None,
    ) -> Sequence[Task]:
        """Notifies the Workload of the completion of a task.

        Args:
            task: The task that finished its execution.
            finish_time: The time at which the task finished its execution.

        Returns:
            A sequence of `Task`s that are released upon the successful completion
            of the given task.
        """
        if task.task_graph not in self._task_graphs:
            raise ValueError(
                f"The TaskGraph {task.task_graph} was not found in the Workload."
            )

        task_graph = self._task_graphs[task.task_graph]
        new_tasks = task_graph.notify_task_completion(task, finish_time)
        if len(new_tasks) == 0 and task_graph.is_complete():
            # The TaskGraph has finished execution, it is safe to remove
            # it from the Workload at this moment.
            tardiness = (
                EventTime(0, EventTime.Unit.US)
                if task_graph.deadline > finish_time
                else finish_time - task_graph.deadline
            )
            self._logger.info(
                f"[{finish_time.time}] Finished the execution of TaskGraph "
                f"{task.task_graph} with the deadline {task_graph.deadline}. "
                f"The tardiness was {tardiness}."
            )
            if csv_logger:
                csv_logger.info(
                    f"{finish_time.time},TASK_GRAPH_FINISH,{task.task_graph},"
                    f"{task_graph.deadline.to(EventTime.Unit.US).time},"
                    f"{tardiness.to(EventTime.Unit.US).time}"
                )
            del self._task_graphs[task.task_graph]
        return new_tasks

    def release_tasks(self, time: Optional[EventTime] = None) -> Sequence[Task]:
        """Requests the Workload to release the set of `Task`s corresponding
        to each `TaskGraph`.

        Args:
            time: The time at which the `Task`s were requested for release.

        Returns:
            A sequence of `Task`s released from the set of `TaskGraph`s.
        """
        released_tasks = []
        for task_graph in self._task_graphs.values():
            released_tasks.extend(task_graph.release_tasks(time))
        return released_tasks

    def get_schedulable_tasks(
        self,
        time: EventTime,
        lookahead: EventTime = EventTime(0, EventTime.Unit.US),
        preemption: bool = False,
        worker_pools: "WorkerPools" = None,  # noqa: F821
    ) -> Sequence[Task]:
        schedulable_tasks = []
        for task_graph in self._task_graphs.values():
            schedulable_tasks.extend(
                task_graph.get_schedulable_tasks(
                    time, lookahead, preemption, worker_pools
                )
            )
        return schedulable_tasks

    def filter(self, function: Callable[[Task], bool]) -> Sequence[Task]:
        """Retrieves the tasks from all the TaskGraphs that return `True`
        on the given function.

        Args:
            function: The function to execute for each of the task.

        Returns:
            A sequence of tasks that return True on the given function.
        """
        filtered_tasks = []
        for task_graph in self._task_graphs.values():
            filtered_tasks.extend(task_graph.filter(function))
        return filtered_tasks

    def __len__(self) -> int:
        """Returns the total number of Tasks in the Workload."""
        total_tasks = 0
        for task_graph in self._task_graphs.values():
            total_tasks += len(task_graph)
        return total_tasks
