from typing import Callable, Mapping, Optional, Sequence, Set, Tuple

import absl

from utils import EventTime, setup_logging
from workload import BranchPredictionPolicy

from .jobs import JobGraph
from .profile import WorkProfile
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
        work_profiles: Optional[Set[WorkProfile]] = None,
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
                "Workload",
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = setup_logging("Workload")

        # Prioritize the creation of the Workload from a TaskGraph
        # for backward compatibility with older versions of logs.
        self._work_profiles = work_profiles
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
        work_profiles = set()
        for job_graph in job_graphs.values():
            for job in job_graph.get_nodes():
                if job.profile is not None:
                    work_profiles.add(job.profile)
        return Workload(
            job_graphs=job_graphs, work_profiles=work_profiles, _flags=_flags
        )

    @staticmethod
    def from_task_graphs(
        task_graphs: Mapping[str, TaskGraph], _flags: Optional["absl.flags"] = None
    ) -> "Workload":
        """Creates a Workload from the given TaskGraphs.

        Args:
            task_graphs: A mapping from the name of the application to its TaskGraph.
        """
        work_profiles = set()
        for task_graph_name, task_graph in task_graphs.items():
            assert (
                task_graph_name == task_graph.name
            ), "Naming mismatch between TaskGraph and Workload."
            for task in task_graph.get_nodes():
                if task.profile is not None:
                    work_profiles.add(task.profile)
        return Workload(
            task_graphs=task_graphs, work_profiles=work_profiles, _flags=_flags
        )

    @staticmethod
    def empty(_flags: Optional["absl.flags"] = None) -> "Workload":
        """Creates an empty Workload.

        Returns:
            An empty `Workload` that has no TaskGraphs.
        """
        return Workload.from_task_graphs(task_graphs={}, _flags=_flags)

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

    def add_job_graphs(
        self, new_job_graphs: Sequence[JobGraph], completion_time: EventTime
    ) -> None:
        """
        Adds the given JobGraphs to the Workload. The task graph of the workload is also
        populated with new tasks generated from new_job_graphs.

        Args:
            new_job_graphs: A sequence of job graphs to be added to the workload.
        """
        self._job_graphs |= {job.name: job for job in new_job_graphs}
        for job_graph in new_job_graphs:
            self._task_graphs |= job_graph.generate_task_graphs(
                completion_time, _flags=self._flags
            )

    def add_task_graph(self, task_graph: TaskGraph) -> None:
        """Adds a single TaskGraph to the Workload.

        Args:
            task_graph: The TaskGraph to be added to the Workload.
        """
        self._task_graphs[task_graph.name] = task_graph

    def add_task_graphs(self, task_graphs: Sequence[TaskGraph]) -> None:
        """Adds multiple TaskGraphs to the Workload.

        Args:
            task_graphs: The TaskGraphs to be added to the Workload.
        """
        self._task_graphs |= {task_graph.name: task_graph for task_graph in task_graphs}

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
    ) -> Tuple[Sequence[Task], Sequence[Task]]:
        """Notifies the Workload of the completion of a task.

        Args:
            task: The task that finished its execution.
            finish_time: The time at which the task finished its execution.

        Returns:
            A tuple of sequence of `Task`s that are released upon the successful
            completion of the given task, and the tasks that were cancelled as a
            result of this successful completion.
        """
        if task.task_graph not in self._task_graphs:
            raise ValueError(
                f"The TaskGraph {task.task_graph} was not found in the Workload."
            )

        task_graph = self._task_graphs[task.task_graph]
        return task_graph.notify_task_completion(task, finish_time)

    def notify_task_graph_completion(
        self, task_graph: TaskGraph, finish_time: EventTime
    ) -> Sequence[Task]:
        """Notifies the Workload of the completion of a task graph.

        Args:
            task_graph: The task graph that finished its execution.
            finish_time: The time at which the task graph finished its execution.

        Returns:
            The tasks from the next task graph to be scheduled, if any.
        """
        if task_graph.name not in self._task_graphs:
            raise ValueError(
                f"The TaskGraph {task_graph} was not found in the Workload."
            )

        job_graph_release_policy = task_graph.job_graph.release_policy
        if (
            job_graph_release_policy
            and job_graph_release_policy.policy_type
            == JobGraph.ReleasePolicyType.CLOSED_LOOP
        ):
            task_graph = task_graph.job_graph.get_next_task_graph(
                start_time=finish_time + EventTime(1, EventTime.Unit.US),
                _flags=self._flags,
            )
            if task_graph is not None:
                self._task_graphs[task_graph.name] = task_graph
                return task_graph.get_releasable_tasks()
            else:
                return []
        return []

    def get_releasable_tasks(self) -> Sequence[Task]:
        """Retrieves the set of Tasks that are ready to run, and can be released
        from each `TaskGraph` in the `Workload`.

        Returns:
            A sequence of `Task`s released from the set of `TaskGraph`s.
        """
        released_tasks = []
        for task_graph in self._task_graphs.values():
            released_tasks.extend(task_graph.get_releasable_tasks())
        return released_tasks

    def get_schedulable_tasks(
        self,
        time: EventTime,
        lookahead: EventTime = EventTime.zero(),
        preemption: bool = False,
        retract_schedules: bool = False,
        worker_pools: "WorkerPools" = None,  # noqa: F821
        policy: BranchPredictionPolicy = BranchPredictionPolicy.ALL,
        branch_prediction_accuracy: float = 0.50,
        release_taskgraphs: bool = False,
        debug: bool = False,
        scheduler_is_task_type: bool = False,
    ) -> Sequence[Task]:
        """Retrieves all the tasks expected to be released within the scheduling
        horizon defined by `time + lookahead`.

        Args:
            time (`EventTime`): The time at which the schedulable tasks are being
                retrieved (the current simulator time).
            lookahead (`EventTime`): The added time upto which the schedulable tasks
                need to be retrieved (i.e. time + lookahead).
            preemption (`bool`): If `True`, ddd currently running tasks to the set of
                schedulable tasks.
            retract_schedules (`bool`): If `True`, allow already scheduled tasks to be
                reconsidered for schedulability.
            worker_pools (`WorkerPools`): A representation of the `WorkerPools` that
                are currently being used for scheduling. This object is only used if
                preemption is enabled.
            policy (`BranchPredictionPolicy`): The branch prediction policy to use when
                deciding what tasks can be considered in the scheduling horizon.
            branch_prediction_accuracy (`float`): The accuracy with which the branches
                should be correctly predicted by the TaskGraph.
            release_taskgraphs (`bool`): If `True`, all tasks of a TaskGraph are made
                available for scheduling if any task in the TaskGraph falls within the
                scheduler lookahead.
            debug (`bool`): If `True`, print debug information about the decision
                making.

        Returns:
            A list of tasks that are schedulable in the `time + lookahead` horizon.
        """
        schedulable_tasks = []
        for task_graph in self._task_graphs.values():
            schedulable_tasks.extend(
                task_graph.get_schedulable_tasks(
                    time,
                    lookahead,
                    preemption,
                    retract_schedules,
                    worker_pools,
                    policy,
                    branch_prediction_accuracy,
                    release_taskgraphs,
                    debug,
                    scheduler_is_task_type,
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

    def get_cancelled_task_graphs(self) -> Sequence[TaskGraph]:
        """Retrieves the TaskGraphs that have been cancelled and will not
        finish execution.

        Returns:
            A (possibly empty) sequence of `TaskGraph`s that have atleast
            one of their tasks in the `CANCELLED` state.
        """
        cancelled_task_graphs = []
        for task_graph in self._task_graphs.values():
            if task_graph.is_cancelled():
                self._logger.debug("The TaskGraph %s was cancelled.", task_graph.name)
                cancelled_task_graphs.append(task_graph)
            elif (
                task_graph.is_complete()
                and task_graph.completion_time > task_graph.deadline
            ):
                self._logger.debug(
                    "The TaskGraph %s was not cancelled, but missed its deadline %s "
                    "since it completed at %s.",
                    task_graph.name,
                    task_graph.deadline,
                    task_graph.completion_time,
                )
            elif task_graph.is_complete():
                self._logger.debug(
                    "The TaskGraph %s completed at %s, and met its deadline of %s.",
                    task_graph.name,
                    task_graph.completion_time,
                    task_graph.deadline,
                )
            else:
                self._logger.debug(
                    "The TaskGraph %s was not cancelled. "
                    "Its sinks are in the state %s.",
                    task_graph.name,
                    [task.state for task in task_graph.get_sink_tasks()],
                )
        return cancelled_task_graphs

    def __len__(self) -> int:
        """Returns the total number of Tasks in the Workload."""
        total_tasks = 0
        for task_graph in self._task_graphs.values():
            total_tasks += len(task_graph)
        return total_tasks

    @property
    def job_graphs(self) -> Mapping[str, JobGraph]:
        """Retrieve the JobGraph instances stored in this Workload."""
        return self._job_graphs

    @property
    def task_graphs(self) -> Mapping[str, TaskGraph]:
        """Retrieve the TaskGraph instances stored in this Workload."""
        return self._task_graphs

    @property
    def work_profiles(self) -> Set[WorkProfile]:
        """Retrieve the `WorkProfile`s that constitute this `Workload`."""
        return self._work_profiles

    def __str__(self) -> str:
        """Returns a string representation of the Workload."""
        task_graph_names = [
            task_graph_name for task_graph_name in self._task_graphs.keys()
        ]
        return f"Workload([{', '.join(task_graph_names)}])"
