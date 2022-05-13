import csv
import json
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
        intended_release_time: int,
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
        self.intended_release_time = intended_release_time
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

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.id == other.id


Resource = namedtuple("Resource", ["name", "id", "quantity"])
WorkerPool = namedtuple("WorkerPool", ["name", "id", "resources"])
WorkerPoolStats = namedtuple(
    "WorkerPoolStats", ["simulator_time", "resource_utilizations"]
)
Scheduler = namedtuple(
    "Scheduler",
    [
        "start_time",
        "end_time",
        "runtime",
        "released_tasks",
        "previously_placed_tasks",
        "total_tasks",
        "placed_tasks",
        "unplaced_tasks",
        "instance_id",
    ],
)

# Types for Events in the system.
SimulatorStart = namedtuple("SimulatorStart", ["start_time", "total_tasks"])
SimulatorEnd = namedtuple(
    "SimulatorEnd", ["end_time", "finished_tasks", "missed_deadlines"]
)
TaskRelease = namedtuple("TaskRelease", ["simulator_time", "task"])
TaskPlacement = namedtuple(
    "TaskPlacement", ["simulator_time", "task", "worker_pool", "resources"]
)
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

        self._events = {}
        self._worker_pools = {}

        self.parse_events(readings)

    def parse_events(self, readings: Mapping[str, Sequence[str]]):
        """Create a sequence of Event tuples from the data read from the CSV
        file.

        Args:
            readings (`Sequence[Sequence[str]]`): The readings retrieved from
                the CSV file.
        """
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
                        task_id=uuid.UUID(reading[8]),
                        intended_release_time=int(reading[4]),
                        release_time=int(reading[5]),
                        runtime=int(reading[6]),
                        deadline=int(reading[7]),
                    )
                    tasks_memo[reading[8]] = task
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
                    resources = [
                        Resource(*reading[i : i + 3]) for i in range(4, len(reading), 3)
                    ]
                    worker_pool_memo[reading[3]] = WorkerPool(
                        name=reading[2],
                        id=uuid.UUID(reading[3]),
                        resources=resources,
                    )
                elif reading[1] == "TASK_PLACEMENT":
                    # Update the task's start time (if needed)
                    task = tasks_memo[reading[4]]
                    simulator_time = int(reading[0])
                    if task.start_time == -1:
                        task.start_time = simulator_time

                    resources = [
                        Resource(*reading[i : i + 3]) for i in range(6, len(reading), 3)
                    ]
                    events.append(
                        TaskPlacement(
                            simulator_time=simulator_time,
                            task=tasks_memo[reading[4]],
                            worker_pool=worker_pool_memo[reading[5]],
                            resources=resources,
                        )
                    )
                else:
                    continue
            self._events[csv_path] = events
            self._worker_pools[csv_path] = worker_pool_memo

    def get_scheduler_invocations(self, csv_path: str) -> Sequence[Scheduler]:
        """Retrieves a sequence of Scheduler invocations from the CSV.

        Args:
            csv_path (`str`): The path to the CSV file whose scheduler invocations
                need to be retrieved.

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
        for index, (scheduler_start, scheduler_finish) in enumerate(
            zip(scheduler_events[::2], scheduler_events[1::2]), start=1
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
                    released_tasks=scheduler_start.released_tasks,
                    previously_placed_tasks=scheduler_start.placed_tasks,
                    total_tasks=scheduler_start.released_tasks
                    + scheduler_start.placed_tasks,
                    placed_tasks=scheduler_finish.placed_tasks,
                    unplaced_tasks=scheduler_finish.unplaced_tasks,
                    instance_id=index,
                )
            )
        return scheduler_invocation_events

    def get_worker_pools(self, csv_path: str) -> Sequence[WorkerPool]:
        """Retrieves the details of the WorkerPool for the given execution.

        Args:
            csv_path (`str`): The path to the CSV file whose worker pool
                needs to be retrieved.

        Returns:
            A `Sequence[WorkerPool]` that depicts the total resources of the WorkerPools
            used in this execution.
        """
        return self._worker_pools[csv_path]

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

    def to_chrome_trace(
        self,
        csv_path: str,
        scheduler_label: str,
        output_path: str,
        at_time: Optional[int] = None,
        trace_fmt: str = "task",
    ):
        """Converts the CSV of the events in a simulation execution to a Chrome tracer
        format.

        Args:
            csv_path (str): The path to the CSV to be converted to Chrome trace.
            output_path (str): The path where the Chrome trace file should be output.
            scheduler_label (str): The name of the scheduler that produced the trace.
            at_time (Optional[int]): Visualize only the tasks that had activity at the
                given time.
            trace_fmt (str): The format of trace to output (task / resource).
        """
        trace = {
            "traceEvents": [],
            "otherData": {"csv_path": csv_path, "scheduler": scheduler_label},
        }

        # Output all the scheduler events.
        for scheduler_event in self.get_scheduler_invocations(csv_path):
            if (
                at_time
                and scheduler_event.start_time
                <= at_time
                <= scheduler_event.start_time + scheduler_event.runtime
            ):
                previously_placed_tasks = scheduler_event.previously_placed_tasks
                trace_event = {
                    "name": f"{scheduler_label}::{scheduler_event.instance_id}",
                    "cat": "scheduler",
                    "ph": "X",
                    "ts": scheduler_event.start_time,
                    "dur": scheduler_event.runtime,
                    "pid": scheduler_label,
                    "tid": "main",
                    "args": {
                        "released_tasks": scheduler_event.released_tasks,
                        "previously_placed_tasks": previously_placed_tasks,
                        "total_tasks": scheduler_event.total_tasks,
                        "placed_tasks": scheduler_event.placed_tasks,
                        "unplaced_tasks": scheduler_event.unplaced_tasks,
                    },
                }
                trace["traceEvents"].append(trace_event)

        if trace_fmt == "resource":
            resource_ids_to_canonical_names = {}
            for worker_pool in self.get_worker_pools(csv_path).values():
                resource_counter = defaultdict(int)
                for resource in worker_pool.resources:
                    resource_counter[resource.name] += 1
                    resource_ids_to_canonical_names[
                        resource.id
                    ] = f"{resource.name}_{resource_counter[resource.name]}"
            task_to_wp_resources = {
                task_placement.task: (
                    task_placement.worker_pool,
                    task_placement.resources,
                )
                for task_placement in self.get_task_placements(csv_path)
            }
        # Output all the tasks.
        for task in self.get_tasks(csv_path):
            # Do not output the tasks if it does not fall within the given time.
            if at_time and not (task.start_time <= at_time <= task.completion_time):
                continue
            if trace_fmt == "task":
                if "." in task.name:
                    # pid = operator name, tid = callback name
                    pid, tid = task.name.split(".", 1)
                else:
                    pid = tid = task.name
                trace_event = {
                    "name": f"{task.name}::{task.timestamp}",
                    "cat": "task,duration",
                    "ph": "X",
                    "ts": task.start_time,
                    "dur": task.completion_time - task.start_time,
                    "pid": pid,
                    "tid": tid,
                    "args": {
                        "name": task.name,
                        "id": str(task.id),
                        "timestamp": task.timestamp,
                        "release_time": task.release_time,
                        "runtime": task.runtime,
                        "deadline": task.deadline,
                        "start_time": task.start_time,
                        "completion_time": task.completion_time,
                        "missed_deadline": task.missed_deadline,
                    },
                }
                trace["traceEvents"].append(trace_event)
            elif trace_fmt == "resource":
                pid = task_to_wp_resources[task][0].name
                tids = [
                    resource_ids_to_canonical_names[resource.id]
                    for resource in task_to_wp_resources[task][1]
                ]
                for tid in tids:
                    trace_event = {
                        "name": f"{task.name}::{task.timestamp}",
                        "cat": "task,duration",
                        "ph": "X",
                        "ts": task.start_time,
                        "dur": task.completion_time - task.start_time,
                        "pid": pid,
                        "tid": tid,
                        "args": {
                            "name": task.name,
                            "id": str(task.id),
                            "timestamp": task.timestamp,
                            "release_time": task.release_time,
                            "runtime": task.runtime,
                            "deadline": task.deadline,
                            "start_time": task.start_time,
                            "completion_time": task.completion_time,
                            "missed_deadline": task.missed_deadline,
                        },
                    }
                    trace["traceEvents"].append(trace_event)
            else:
                raise ValueError(f"Undefined execution mode: {trace_fmt}")

        # Output all the missed deadlines.
        for missed_deadline_event in self.get_missed_deadline_events(csv_path):
            task = missed_deadline_event.task
            # Do not output the tasks if it does not fall within the given time.
            if at_time and not (task.start_time <= at_time <= task.completion_time):
                continue
            if trace_fmt == "task":
                if "." in task.name:
                    # pid = operator name, tid = callback name
                    pid, tid = task.name.split(".", 1)
                else:
                    pid = tid = task.name
                trace_event = {
                    "name": f"{task.name}::{task.timestamp}",
                    "cat": "task,missed,deadline,instant",
                    "ph": "i",
                    "ts": task.deadline,
                    "pid": pid,
                    "tid": tid,
                    "s": "t",  # The scope of the missed deadline events is per thread.
                }
                trace["traceEvents"].append(trace_event)
            elif trace_fmt == "resource":
                pid = task_to_wp_resources[task][0].name
                tids = [
                    resource_ids_to_canonical_names[resource.id]
                    for resource in task_to_wp_resources[task][1]
                ]
                for tid in tids:
                    trace_event = {
                        "name": f"{task.name}::{task.timestamp}",
                        "cat": "task,missed,deadline,instant",
                        "ph": "i",
                        "ts": task.deadline,
                        "pid": pid,
                        "tid": tid,
                        # The scope of the missed deadline events is per thread.
                        "s": "t",
                    }
                    trace["traceEvents"].append(trace_event)
            else:
                raise ValueError(f"Undefined execution mode: {trace_fmt}")

        with open(output_path, "w") as f:
            json.dump(trace, f, indent=4, sort_keys=True)
