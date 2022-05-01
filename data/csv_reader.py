import csv
import uuid
from collections import defaultdict, namedtuple
from functools import total_ordering
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401


# Types for Objects in the simulation.
@total_ordering
class Task(object):
    def __init__(
        self,
        name: str,
        timestamp: int,
        task_id: uuid.UUID,
        release_time: int,
        runtime: int,
        deadline: int,
        start_time: int = -1,
        completion_time: int = -1,
        missed_deadline: bool = False,
    ):
        self.name = name
        self.timestamp = timestamp
        self.id = task_id
        # All times are in microseconds.
        self.release_time = release_time
        self.runtime = runtime
        self.deadline = deadline
        self.start_time = start_time
        self.completion_time = completion_time
        self.missed_deadline = missed_deadline

    def __str__(self):
        return f"Task(name={self.name}, timestamp={self.timestamp})"

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if self == other:
            return False
        if self.timestamp > other.timestamp:
            return False
        elif self.timestamp == other.timestamp:
            if self.release_time > other.release_time:
                return False
            elif self.release_time == other.release_time:
                return self.runtime <= other.runtime
        return True

    def __eq__(self, other):
        return self.id == other.id


WorkerPool = namedtuple("WorkerPool", ["name", "id"])
WorkerPoolStats = namedtuple(
    "WorkerPoolStats", ["simulator_time", "resource_utilizations"]
)
Scheduler = namedtuple(
    "Scheduler",
    [
        "start_time",
        "end_time",
        "runtime",
        "total_tasks",
        "placed_tasks",
        "unplaced_tasks",
    ],
)

# Types for Events in the system.
SimulatorStart = namedtuple("SimulatorStart", ["start_time", "total_tasks"])
SimulatorEnd = namedtuple(
    "SimulatorEnd", ["end_time", "finished_tasks", "missed_deadlines"]
)
TaskRelease = namedtuple("TaskRelease", ["simulator_time", "task"])
TaskPlacement = namedtuple("TaskPlacement", ["simulator_time", "task", "worker_pool"])
TaskFinished = namedtuple("TaskFinished", ["simulator_time", "task"])
MissedDeadline = namedtuple("MissedDeadline", ["simulator_time", "task"])
SchedulerStart = namedtuple(
    "SchedulerStart", ["simulator_time", "released_tasks", "placed_tasks"]
)
SchedulerFinished = namedtuple(
    "SchedulerFinished", ["simulator_time", "runtime", "placed_tasks", "unplaced_tasks"]
)
WorkerPoolUtilization = namedtuple(
    "WorkerPoolUtilization",
    ["simulator_time", "resource_name", "allocated_quantity", "available_quantity"],
)


class CSVReader(object):
    """Reads the data from the CSV logs of a Simulator run.

    Args:
        csv_paths (`Sequence[str]`): The paths to the CSVs where the results
            are stored.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self, csv_paths: str, _flags: Optional["absl.flags"] = None):
        readings = {}
        for csv_path in csv_paths:
            with open(csv_path, "r") as csv_file:
                path_readings = []
                for line in csv.reader(csv_file):
                    path_readings.append(line)
                readings[csv_path] = path_readings

        self._events = CSVReader.parse_events(readings)

    @staticmethod
    def parse_events(
        readings: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Sequence[tuple]]:
        """Create a sequence of Event tuples from the data read from the CSV
        file.

        Args:
            readings (`Sequence[Sequence[str]]`): The readings retrieved from
                the CSV file.

        Returns:
            A mapping of CSV name to the sequence of Events converted from the
            CSV data.
        """
        parsed_events = {}
        for csv_path, csv_readings in readings.items():
            events = []
            tasks_memo = {}
            worker_pool_memo = {}
            for reading in csv_readings:
                if reading[1] == "SIMULATOR_START":
                    events.append(
                        SimulatorStart(
                            start_time=int(reading[0]), total_tasks=int(reading[2])
                        )
                    )
                elif reading[1] == "SIMULATOR_END":
                    events.append(
                        SimulatorEnd(
                            end_time=int(reading[0]),
                            finished_tasks=int(reading[2]),
                            missed_deadlines=int(reading[3]),
                        )
                    )
                elif reading[1] == "TASK_RELEASE":
                    task = Task(
                        name=reading[2],
                        timestamp=int(reading[3]),
                        task_id=uuid.UUID(reading[7]),
                        release_time=int(reading[4]),
                        runtime=int(reading[5]),
                        deadline=int(reading[6]),
                    )
                    tasks_memo[reading[7]] = task
                    events.append(
                        TaskRelease(simulator_time=int(reading[0]), task=task)
                    )
                elif reading[1] == "TASK_FINISHED":
                    task = tasks_memo[reading[6]]
                    task.completion_time = int(reading[4])
                    events.append(
                        TaskFinished(simulator_time=int(reading[0]), task=task)
                    )
                elif reading[1] == "MISSED_DEADLINE":
                    task = tasks_memo[reading[5]]
                    task.missed_deadline = True
                    events.append(
                        MissedDeadline(simulator_time=int(reading[0]), task=task)
                    )
                elif reading[1] == "SCHEDULER_START":
                    events.append(
                        SchedulerStart(
                            simulator_time=int(reading[0]),
                            released_tasks=int(reading[2]),
                            placed_tasks=int(reading[3]),
                        )
                    )
                elif reading[1] == "SCHEDULER_FINISHED":
                    events.append(
                        SchedulerFinished(
                            simulator_time=int(reading[0]),
                            runtime=int(reading[2]),
                            placed_tasks=int(reading[3]),
                            unplaced_tasks=int(reading[4]),
                        )
                    )
                elif reading[1] == "WORKER_POOL_UTILIZATION":
                    events.append(
                        WorkerPoolUtilization(
                            simulator_time=int(reading[0]),
                            resource_name=reading[2],
                            allocated_quantity=float(reading[3]),
                            available_quantity=float(reading[4]),
                        )
                    )
                elif reading[1] == "WORKER_POOL":
                    worker_pool_memo[reading[3]] = WorkerPool(
                        name=reading[2], id=uuid.UUID(reading[3])
                    )
                elif reading[1] == "TASK_PLACEMENT":
                    events.append(
                        TaskPlacement(
                            simulator_time=int(reading[0]),
                            task=tasks_memo[reading[4]],
                            worker_pool=worker_pool_memo[reading[5]],
                        )
                    )
                else:
                    continue
            parsed_events[csv_path] = events
        return parsed_events

    def get_scheduler_invocations(self, csv_path: str) -> Sequence[Scheduler]:
        """Retrieves a sequence of Scheduler invocations from the CSV.

        Args:
            csv_path (`str`): The path to the CSV file whose worker pool
                statistics need to be retrieved.

        Returns:
            A `Sequence[Scheduler]` that depicts the number of placed, unplaced
            and total tasks, along with the runtime of the invocation.
        """
        scheduler_events = []
        for event in self._events[csv_path]:
            if type(event) == SchedulerStart or type(event) == SchedulerFinished:
                scheduler_events.append(event)

        # Form scheduler invocation events from the retrieved events.
        scheduler_invocation_events = []
        for scheduler_start, scheduler_finish in zip(
            scheduler_events[::2], scheduler_events[1::2]
        ):
            assert (
                type(scheduler_start) == SchedulerStart
            ), "Incorrect type found for scheduler_start event: {}".format(
                type(scheduler_start)
            )
            assert (
                type(scheduler_finish) == SchedulerFinished
            ), "Incorrect type found for scheduler_finish event: {}".format(
                type(scheduler_finish)
            )
            scheduler_invocation_events.append(
                Scheduler(
                    start_time=scheduler_start.simulator_time,
                    end_time=scheduler_finish.simulator_time,
                    runtime=scheduler_finish.runtime,
                    total_tasks=scheduler_start.released_tasks
                    + scheduler_start.placed_tasks,
                    placed_tasks=scheduler_finish.placed_tasks,
                    unplaced_tasks=scheduler_finish.unplaced_tasks,
                )
            )
        return scheduler_invocation_events

    def get_worker_pool_utilizations(self, csv_path: str) -> Sequence[WorkerPoolStats]:
        """Retrieves the statistics of the utilization of the WorkerPool at
        different points in time.

        Args:
            csv_path (`str`): The path to the CSV file whose worker pool
                statistics need to be retrieved.

        Returns:
            A `Sequence[WorkerPoolStats]` that depicts the usage of resources
            across all the WorkerPools at each invocation of the scheduler.
        """
        worker_pool_utilizations = defaultdict(list)
        for event in self._events[csv_path]:
            if type(event) == WorkerPoolUtilization:
                worker_pool_utilizations[event.simulator_time].append(event)

        # Order the utilizations and construct a stats object.
        worker_pool_stats = []
        for simulator_time in sorted(worker_pool_utilizations.keys()):
            utilizations = worker_pool_utilizations[simulator_time]
            resource_utilizations = {}
            for utilization in utilizations:
                resource_utilizations[utilization.resource_name] = (
                    utilization.allocated_quantity,
                    utilization.available_quantity,
                )
            worker_pool_stats.append(
                WorkerPoolStats(
                    simulator_time=simulator_time,
                    resource_utilizations=resource_utilizations,
                )
            )
        return worker_pool_stats

    def get_tasks(self, csv_path: str) -> Sequence[Task]:
        """Retrieves the tasks ordered by their release time.

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `Sequence[Task]` that depicts the tasks in the execution,
            ordered by their release time.
        """
        tasks = []
        for event in self._events[csv_path]:
            if type(event) == TaskRelease:
                tasks.append(event.task)
        return tasks

    def get_task_placements(self, csv_path: str) -> Sequence[TaskPlacement]:
        """Retrives the task placements events.

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `Sequence[TaskPlacement]` that contains the task placements,
            ordered by their placement time.
        """
        task_placements = []
        for event in self._events[csv_path]:
            if type(event) == TaskPlacement:
                task_placements.append(event)
        return task_placements

    def get_simulator_end_time(self, csv_path: str) -> int:
        """Retrieves the time at which the simulator ended.

        Args:
            csv_path (`str`): The path to the CSV file whose end time needs to
                be retrieved.

        Returns:
            The end time of the simulation of the given CSV file.
        """
        for event in self._events[csv_path]:
            if type(event) == SimulatorEnd:
                return event.end_time
        raise ValueError("No SIMULATOR_END event found in the logs.")

    def get_missed_deadline_events(self, csv_path: str) -> Sequence[MissedDeadline]:
        """Retrieves the missed deadline events.

        Args:
            csv_path (`str`): The path to the CSV file whose missed deadline
            events need to be retrieved.

        Returns:
            A sequence of missed deadline events for the given CSV file.
        """
        missed_deadline_events = []
        for event in self._events[csv_path]:
            if type(event) == MissedDeadline:
                missed_deadline_events.append(event)
        return missed_deadline_events
