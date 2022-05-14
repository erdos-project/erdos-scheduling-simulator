import csv
import json
import uuid
from collections import defaultdict, namedtuple
from functools import total_ordering
from operator import attrgetter
from typing import Mapping, Optional, Sequence, Tuple, Union

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
    ):
        self.name = name
        self.timestamp = timestamp
        self.id = task_id
        # All times are in microseconds.
        self.intended_release_time = intended_release_time
        self.release_time = release_time
        self.runtime = runtime
        self.deadline = deadline

        # Values updated from the TASK_PLACEMENT event.
        self.was_placed = False
        self.start_time = None
        self.placement_time = None
        self.placed_on_worker_pool = None
        self.resources_used = None

        # Values updated from the TASK_SKIP event.
        self.skipped_times = []

        # Values updated from the TASK_FINISHED event.
        self.completion_time = None

        # Values updated from the MISSED_DEADLINE event.
        self.missed_deadline = False
        self.deadline_miss_detected_at = None

    def get_deadline_delay(self) -> int:
        """Retrieve the deadline delay in microseconds.

        If a positive value is returned, the deadline was missed and if a negative
        value is returned, the deadline was not missed.

        Returns:
            The time between the completion of the task and the deadline.
        """
        deadline_delay = self.completion_time - self.deadline
        assert (deadline_delay > 0) == self.missed_deadline, (
            f"Inconsistency between the deadline delay ({deadline_delay}) and "
            f"the missed deadline flag ({self.missed_deadline}."
        )
        return deadline_delay

    def get_release_delay(self) -> Optional[int]:
        """Retrieve the release delay of the task.

        If no intended release time is found, the method returns None.

        Returns:
            The time between the intended and the actual release of the task.
        """
        if self.intended_release_time != -1:
            return self.release_time - self.intended_release_time

    def get_placement_delay(self) -> int:
        """Retrieve the placement delay in microseconds.

        Returns:
            The time between the placement of the task and its release.
        """
        assert self.was_placed, f"The task {self} was never placed."
        return self.placement_time - self.release_time

    def update_placement(self, csv_reading: str):
        """Updates the values of the Task based on the TASK_PLACEMENT event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_PLACEMENT`.
        """
        assert (
            csv_reading[1] == "TASK_PLACEMENT"
        ), f"The event {csv_reading[1]} was not of type TASK_PLACEMENT."
        assert self.was_placed == False, f"The task {self} was already placed."
        placement_time = int(csv_reading[0])
        self.start_time = self.placement_time = placement_time
        self.placed_on_worker_pool = uuid.UUID(csv_reading[5])
        self.resources_used = [
            Resource(*csv_reading[i : i + 3]) for i in range(6, len(csv_reading), 3)
        ]
        self.was_placed = True

    def update_skip(self, csv_reading: str):
        """Updates the values of the Task based on the TASK_SKIP event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_SKIP`.
        """
        assert (
            csv_reading[1] == "TASK_SKIP"
        ), f"The event {csv_reading[1]} was not of type TASK_SKIP."
        self.skipped_times.append(int(csv_reading[0]))

    def update_finish(self, csv_reading: str):
        """Updates the values of the Task based on the TASK_FINISHED event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_FINISHED`.
        """
        assert (
            csv_reading[1] == "TASK_FINISHED"
        ), f"The event {csv_reading[1]} was not of type TASK_FINISHED."
        self.completion_time = int(csv_reading[4])

    def update_missed_deadline(self, csv_reading: str):
        """Updates the values of the Task based on the MISSED_DEADLINE event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `MISSED_DEADLINE`.
        """
        assert (
            csv_reading[1] == "MISSED_DEADLINE"
        ), f"The event {csv_reading[1]} was not of type MISSED_DEADLINE."
        self.missed_deadline = True
        self.deadline_miss_detected_at = int(csv_reading[0])

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
        self._tasks = {}
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
                    tasks_memo[reading[8]] = Task(
                        name=reading[2],
                        timestamp=int(reading[3]),
                        task_id=uuid.UUID(reading[8]),
                        intended_release_time=int(reading[4]),
                        release_time=int(reading[5]),
                        runtime=int(reading[6]),
                        deadline=int(reading[7]),
                    )
                elif reading[1] == "TASK_FINISHED":
                    # Update the task with the completion event data.
                    tasks_memo[reading[6]].update_finish(reading)
                elif reading[1] == "MISSED_DEADLINE":
                    # Update the task with the completion event data.
                    tasks_memo[reading[5]].update_missed_deadline(reading)
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
                    # Update the task with the placement event data.
                    tasks_memo[reading[4]].update_placement(reading)
                elif reading[1] == "TASK_SKIP":
                    # Update the task with the skip data.
                    tasks_memo[reading[4]].update_skip(reading)
                else:
                    continue
            self._events[csv_path] = events
            self._tasks[csv_path] = list(
                sorted(tasks_memo.values(), key=attrgetter("release_time"))
            )
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
        return self._tasks[csv_path]

    def get_tasks_with_placement_issues(self, csv_path: str) -> Sequence[Task]:
        """Retrieves the tasks that had placement issues (i.e., had a TASK_SKIP).

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `Sequence[Task]` that contains the task with placement issues, ordered by
            release time.
        """
        return [
            task for task in self.get_tasks(csv_path) if len(task.skipped_times) > 0
        ]

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

    def to_chrome_trace(
        self,
        csv_path: str,
        scheduler_label: str,
        output_path: str,
        between_time: Union[int, Tuple[int, int]] = None,
        trace_fmt: str = "task",
        show_deadlines: str = "missed",
        with_placement_issues: bool = False,
    ):
        """Converts the CSV of the events in a simulation execution to a Chrome tracer
        format.

        Args:
            csv_path (str): The path to the CSV to be converted to Chrome trace.
            output_path (str): The path where the Chrome trace file should be output.
            scheduler_label (str): The name of the scheduler that produced the trace.
            between_time (Union[int, Tuple[int, int]]): Visualize only the tasks that
                had activity either between the given times or at the given time.
            trace_fmt (str): The format of trace to output (task / resource).
            show_deadlines (str): Choose between ('never', 'missed', 'always') to
                affect which deadlines are shown in the trace.
        """
        trace = {
            "traceEvents": [],
            "otherData": {"csv_path": csv_path, "scheduler": scheduler_label},
        }
        if not isinstance(between_time, int) or (
            isinstance(between_time, Sequence) and len(between_time) != 2
        ):
            raise ValueError(
                "between_time should either be an integer specifying an exact time, "
                "or a tuple specifying an interval."
            )
        if show_deadlines not in ("never", "missed", "always"):
            raise ValueError(
                f"The value of show_deadlines ({show_deadlines}) must be chosen from "
                f"(never, missed, always)."
            )

        def check_if_time_intersects(start_time, end_time):
            if isinstance(between_time, int):
                return start_time <= between_time <= end_time
            elif isinstance(between_time, Sequence):
                return (
                    min(end_time, between_time[1]) - max(start_time, between_time[0])
                ) >= 0
            else:
                return True

        # Output all the scheduler events.
        for scheduler_event in self.get_scheduler_invocations(csv_path):
            if check_if_time_intersects(
                scheduler_event.start_time,
                scheduler_event.start_time + scheduler_event.runtime,
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
            worker_pools = self.get_worker_pools(csv_path)
            for worker_pool in worker_pools.values():
                resource_counter = defaultdict(int)
                for resource in worker_pool.resources:
                    resource_counter[resource.name] += 1
                    resource_ids_to_canonical_names[
                        resource.id
                    ] = f"{resource.name}_{resource_counter[resource.name]}"
            task_to_wp_resources = {
                task: (
                    worker_pools[str(task.placed_on_worker_pool)],
                    task.resources_used,
                )
                for task in self.get_tasks(csv_path)
            }
        # Output all the tasks.
        for task in self.get_tasks(csv_path):
            # Do not output the tasks if it does not fall within the given time.
            if not check_if_time_intersects(task.start_time, task.completion_time):
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

        # Find the tasks that conform to the show_deadlines requirements
        tasks_for_deadline_events = []
        if show_deadlines == "missed":
            tasks_for_deadline_events.extend(
                [task for task in self.get_tasks(csv_path) if task.missed_deadline]
            )
        elif show_deadlines == "always":
            tasks_for_deadline_events.extend(
                [task for task in self.get_tasks(csv_path)]
            )

        # Output all the requested deadlines.
        for task in tasks_for_deadline_events:
            # Do not output the tasks if it does not fall within the given time.
            if not check_if_time_intersects(task.start_time, task.completion_time):
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

        # If placement issues were requested in resource trace, output all the tasks
        # with their actual release time and runtime if the skipped events were within
        # the time frame.
        if trace_fmt == "resource" and with_placement_issues:
            for task in self.get_tasks_with_placement_issues(csv_path):
                if check_if_time_intersects(
                    task.release_time, task.release_time + task.runtime
                ):
                    trace_event = {
                        "name": f"{task.name}::{task.timestamp}",
                        "cat": "task,duration",
                        "ph": "X",
                        "ts": task.release_time,
                        "dur": task.runtime,
                        "pid": "Placement Issues",
                        "tid": task.name,
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

                    # Output the deadline event according to the required strategy.
                    if show_deadlines == "missed" and task.missed_deadline:
                        trace_event = {
                            "name": f"{task.name}::{task.timestamp}",
                            "cat": "task,missed,deadline,instant",
                            "ph": "i",
                            "ts": task.deadline,
                            "pid": "Placement Issues",
                            "tid": task.name,
                            # The scope of the missed deadline events is per thread.
                            "s": "t",
                        }
                        trace["traceEvents"].append(trace_event)
                    elif show_deadlines == "always":
                        trace_event = {
                            "name": f"{task.name}::{task.timestamp}",
                            "cat": "task,missed,deadline,instant",
                            "ph": "i",
                            "ts": task.deadline,
                            "pid": "Placement Issues",
                            "tid": task.name,
                            # The scope of the missed deadline events is per thread.
                            "s": "t",
                        }
                        trace["traceEvents"].append(trace_event)

        with open(output_path, "w") as f:
            json.dump(trace, f, indent=4, sort_keys=True)
