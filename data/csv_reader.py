import csv
import json
from collections import defaultdict
from operator import add, attrgetter
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import absl  # noqa: F401

from data.csv_types import (
    Resource,
    Scheduler,
    Simulator,
    Task,
    TaskGraph,
    WorkerPool,
    WorkerPoolStats,
    WorkerPoolUtilization,
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

        self._simulators: dict[str, Simulator] = {}
        self.parse_events(readings)

    def parse_events(self, readings: Mapping[str, Sequence[str]]):
        """Create a sequence of Event tuples from the data read from the CSV
        file.

        Args:
            readings (`Sequence[Sequence[str]]`): The readings retrieved from
                the CSV file.
        """
        for csv_path, csv_readings in readings.items():
            simulator = None
            tasks: dict[str, Task] = {}
            task_graphs: dict[str, TaskGraph] = {}
            worker_pools = {}
            schedulers: list[Scheduler] = []
            for reading in csv_readings:
                try:
                    # TODO: This
                    if reading[1] == "SIMULATOR_START":
                        simulator = Simulator(
                            csv_path=csv_path,
                            start_time=int(reading[0]),
                        )
                    elif reading[1] == "UPDATE_WORKLOAD":
                        simulator.total_tasks += int(reading[2])
                    elif reading[1] == "LOG_STATS":
                        assert (
                            simulator is not None
                        ), "No SIMULATOR_START found for a corresponding SIMULATOR_END."
                        simulator.update_stats(reading)
                    elif reading[1] == "SIMULATOR_END":
                        assert (
                            simulator is not None
                        ), "No SIMULATOR_START found for a corresponding SIMULATOR_END."
                        simulator.update_finish(reading)
                    elif reading[1] == "TASK_RELEASE":
                        tasks[reading[7]] = Task(
                            name=reading[2],
                            task_graph=reading[8],
                            timestamp=int(reading[3]),
                            task_id=reading[7],
                            intended_release_time=int(reading[4]),
                            release_time=int(reading[5]),
                            deadline=int(reading[6]),
                            window_to_execute=int(reading[6]) - int(reading[5]),
                            slowest_execution_time=int(reading[9]),
                        )
                    elif reading[1] == "TASK_FINISHED":
                        # Update the task with the completion event data.
                        tasks[reading[7]].update_finish(reading)
                    elif reading[1] == "MISSED_DEADLINE":
                        # Update the task with the completion event data.
                        tasks[reading[5]].update_missed_deadline(reading)
                    elif reading[1] == "SCHEDULER_START":
                        schedulers.append(
                            Scheduler(
                                start_time=int(reading[0]),
                                released_tasks=int(reading[2]),
                                previously_placed_tasks=int(reading[3]),
                                instance_id=len(schedulers) + 1,
                            )
                        )
                    elif reading[1] == "SCHEDULER_FINISHED":
                        # Update the Scheduler with the completion event data.
                        schedulers[-1].update_finish(reading)
                    elif reading[1] == "WORKER_POOL_UTILIZATION":
                        worker_pools[reading[2]].utilizations.append(
                            WorkerPoolUtilization(
                                simulator_time=int(reading[0]),
                                resource_name=reading[3],
                                allocated_quantity=float(reading[4]),
                                available_quantity=float(reading[5]),
                            )
                        )
                    elif reading[1] == "WORKER_POOL":
                        resources = [
                            Resource(*reading[i : i + 3])
                            for i in range(4, len(reading), 3)
                        ]
                        worker_pools[reading[3]] = WorkerPool(
                            name=reading[2],
                            id=reading[3],
                            resources=resources,
                        )
                    elif reading[1] == "TASK_PLACEMENT":
                        # Update the task with the placement event data.
                        tasks[reading[5]].update_placement(reading, worker_pools)
                    elif reading[1] == "TASK_CANCEL":
                        # Update the task with the placement event data.
                        if reading[4] not in tasks:
                            tasks[reading[4]] = Task(
                                name=reading[2],
                                task_graph=reading[5],
                                timestamp=int(reading[3]),
                                task_id=reading[4],
                                intended_release_time=None,
                                release_time=None,
                                deadline=None,
                                window_to_execute=None,
                                # Checking if len(reading) > 6
                                # is for backward compatibility
                                slowest_execution_time=(
                                    int(reading[6]) if len(reading) > 6 else None
                                ),
                            )
                        tasks[reading[4]].cancelled = True
                        tasks[reading[4]].cancelled_at = int(reading[0])
                        task_graphs[reading[5]].cancelled = True
                        task_graphs[reading[5]].cancelled_at = int(reading[0])
                    elif reading[1] == "TASK_SKIP":
                        if reading[4] in tasks:
                            # Update the task with the skip data.
                            tasks[reading[4]].update_skip(reading)
                    elif reading[1] == "TASK_PREEMPT":
                        # Update the placement with the preemption time.
                        tasks[reading[4]].update_preempt(reading)
                    elif reading[1] == "TASK_MIGRATED":
                        # Update the placement with the migration time.
                        tasks[reading[4]].update_migration(reading, worker_pools)
                    elif reading[1] == "TASK_SCHEDULED":
                        # Add the task to the last scheduler's invocation.
                        schedulers[-1].update_task_schedule(reading)
                    elif reading[1] == "TASK_GRAPH_RELEASE":
                        task_graphs[reading[4]] = TaskGraph(
                            name=reading[4],
                            release_time=int(reading[2]),
                            deadline=int(reading[3]),
                            num_tasks=int(reading[5]),
                            # Checking if len(reading) > 6
                            # is for backward compatibility
                            critical_path_time=(
                                int(reading[6]) if len(reading) > 6 else None
                            ),
                        )
                    elif reading[1] == "TASK_GRAPH_FINISHED":
                        task_graphs[reading[2]].completion_at = int(reading[0])
                        task_graphs[reading[2]].cancelled = False
                        task_graphs[reading[2]].slack = (
                            task_graphs[reading[2]].deadline
                            - task_graphs[reading[2]].completion_at
                        )
                    elif reading[1] == "MISSED_TASK_GRAPH_DEADLINE":
                        task_graphs[reading[2]].deadline_miss_detected_at = int(
                            reading[0]
                        )
                    elif reading[0] == "input_flag":
                        continue
                    else:
                        print(f"[x] Unknown event type: {reading[1]}")
                except Exception as e:
                    raise ValueError(
                        f"Error while parsing the following line: {reading} "
                        "from {csv_path}"
                    ) from e

            # Some sanity check for task state and task graph state consistency
            canceled_task_graphs_count = len(
                [tg for tg in task_graphs.values() if tg.cancelled]
            )
            missed_deadline_task_graphs_count = len(
                [tg for tg in task_graphs.values() if tg.missed_deadline]
            )
            completed_task_graphs_count = len(
                [tg for tg in task_graphs.values() if tg.was_completed]
            )
            assert simulator.finished_task_graphs == completed_task_graphs_count
            assert simulator.missed_taskgraphs == missed_deadline_task_graphs_count
            assert simulator.dropped_taskgraphs == canceled_task_graphs_count

            simulator.worker_pools = list(worker_pools.values())
            simulator.tasks = list(
                sorted(tasks.values(), key=lambda x: x.release_time_compare_key)
            )
            simulator.scheduler_invocations = schedulers
            simulator.task_graphs = task_graphs
            self._simulators[csv_path] = simulator

    def get_scheduler_invocations(self, csv_path: str) -> Sequence[Scheduler]:
        """Retrieves a sequence of Scheduler invocations from the CSV.

        Args:
            csv_path (`str`): The path to the CSV file whose scheduler invocations
                need to be retrieved.

        Returns:
            A `Sequence[Scheduler]` that depicts the number of placed, unplaced
            and total tasks, along with the runtime of the invocation.
        """
        return self._simulators[csv_path].scheduler_invocations

    def get_worker_pools(self, csv_path: str) -> Sequence[WorkerPool]:
        """Retrieves the details of the WorkerPool for the given execution.

        Args:
            csv_path (`str`): The path to the CSV file whose worker pool
                needs to be retrieved.

        Returns:
            A `Sequence[WorkerPool]` that depicts the total resources of the WorkerPools
            used in this execution.
        """
        return self._simulators[csv_path].worker_pools

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
        for worker_pool in self.get_worker_pools(csv_path):
            for utilization in worker_pool.utilizations:
                worker_pool_utilizations[utilization.simulator_time].append(utilization)

        # Order the utilizations and construct a stats object.
        worker_pool_stats = []
        for simulator_time in sorted(worker_pool_utilizations.keys()):
            utilizations = worker_pool_utilizations[simulator_time]
            resource_utilizations = defaultdict(lambda: (0, 0))
            for utilization in utilizations:
                resource_utilizations[utilization.resource_name] = tuple(
                    map(
                        add,
                        resource_utilizations[utilization.resource_name],
                        (
                            utilization.allocated_quantity,
                            utilization.available_quantity,
                        ),
                    )
                )
            worker_pool_stats.append(
                WorkerPoolStats(
                    simulator_time=simulator_time,
                    resource_utilizations=dict(resource_utilizations),
                )
            )
        return worker_pool_stats

    def get_goodput(self, csv_path: str) -> float:
        return self._simulators[csv_path].goodput_taskgraphs

    def get_tasks(self, csv_path: str) -> Sequence[Task]:
        """Retrieves the tasks ordered by their release time.

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `Sequence[Task]` that depicts the tasks in the execution,
            ordered by their release time.
        """
        return self._simulators[csv_path].tasks

    def get_task_graph(self, csv_path: str) -> dict[str, TaskGraph]:
        """Retrieves the tasks ordered by their release time.

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `Sequence[Task]` that depicts the tasks in the execution,
            ordered by their release time.
        """
        return self._simulators[csv_path].task_graphs

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

    def get_time_spent_on_completed_canceled_miss_deadline_task_graph(
        self, csv_path: str
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Calculate the time scheduler spent on running tasks belong to task
           graphs that
            1. completed
            2. eventually got canceled
            3. missed deadline.

        This helped you identify the wasted time spent on canceled task graph.

        Args:
            csv_path (`str`): The path to the CSV file whose tasks need to
                be retrieved.

        Returns:
            A `tuple[dict[str, int], dict[str, int], dict[str, int]]` where each dict
            contains the task graph name to time spent.
        """
        completed_task_graph_run_time = defaultdict(int)
        canceled_task_graph_run_time = defaultdict(int)
        miss_deadline_task_graph_run_time = defaultdict(int)
        for task in self.get_tasks(csv_path):
            if self.get_task_graph(csv_path)[task.task_graph].cancelled:
                if task.was_completed:
                    canceled_task_graph_run_time[task.task_graph] += task.runtime
                elif task.cancelled and task.placement_time is not None:
                    canceled_task_graph_run_time[task.task_graph] += (
                        task.cancelled_at - task.placement_time
                    )
            else:
                # This task graph was completed
                assert self.get_task_graph(csv_path)[task.task_graph].was_completed
                completed_task_graph_run_time[task.task_graph] += task.runtime
                if self.get_task_graph(csv_path)[task.task_graph].missed_deadline:
                    miss_deadline_task_graph_run_time[task.task_graph] += task.runtime
        return (
            completed_task_graph_run_time,
            canceled_task_graph_run_time,
            miss_deadline_task_graph_run_time,
        )

    def get_simulator_end_time(self, csv_path: str) -> int:
        """Retrieves the time at which the simulator ended.

        Args:
            csv_path (`str`): The path to the CSV file whose end time needs to
                be retrieved.

        Returns:
            The end time of the simulation of the given CSV file.
        """
        return self._simulators[csv_path].end_time

    def to_chrome_trace(
        self,
        csv_path: str,
        scheduler_label: str,
        output_path: str,
        between_time: Union[int, Tuple[int, int]] = None,
        trace_fmt: str = "task",
        show_release_times: str = "never",
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
            trace_fmt (str): The format of trace to output
                (task / resource / taskgraph / application).
            show_release_times (str): Choose between ('never', 'intended', 'always')
                to affect which release times are shown in the trace.
            show_deadlines (str): Choose between ('never', 'missed', 'always') to
                affect which deadlines are shown in the trace.
        """
        if between_time and (
            not isinstance(between_time, int)
            and (isinstance(between_time, Sequence) and len(between_time) != 2)
        ):
            raise ValueError(
                "between_time should either be an integer specifying an exact time, "
                "or a tuple specifying an interval."
            )

        if show_release_times not in ("never", "intended", "always"):
            raise ValueError(
                f"The value of show_release_times ({show_release_times}) must be "
                f"chosen from (never, missed, always)."
            )

        if show_deadlines not in ("never", "missed", "always"):
            raise ValueError(
                f"The value of show_deadlines ({show_deadlines}) must be chosen from "
                f"(never, missed, always)."
            )

        def check_if_time_intersects(between_time, start_time, end_time):
            if isinstance(between_time, int):
                return start_time <= between_time <= end_time
            elif isinstance(between_time, Sequence):
                return (
                    min(end_time, between_time[1]) - max(start_time, between_time[0])
                ) >= 0
            else:
                return True

        trace = {
            "traceEvents": [],
            "otherData": {
                "csv_path": csv_path,
                "scheduler": scheduler_label,
                "between_time": between_time,
                "show_release_times": show_release_times,
                "show_deadlines": show_deadlines,
            },
        }

        # Output all the scheduler events.
        for scheduler_event in self.get_scheduler_invocations(csv_path):
            if scheduler_event.runtime is None:
                # In case the simulation didn't finish the last scheduler run.
                break
            if check_if_time_intersects(
                between_time,
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
                        "num_placed_tasks": scheduler_event.num_placed_tasks,
                        "num_unplaced_tasks": scheduler_event.num_unplaced_tasks,
                    },
                }
                trace["traceEvents"].append(trace_event)

                # Output the scheduler's placement as individual elements, if requested.
                if trace_fmt == "scheduler":
                    for placement in scheduler_event.task_placements:
                        trace_event = {
                            "name": f"{placement.task_name}::{placement.timestamp}",
                            "cat": "scheduler,placement",
                            "ph": "X",
                            "ts": placement.placement_time,
                            "dur": placement.completion_time - placement.placement_time,
                            "pid": f"{scheduler_label}::{scheduler_event.instance_id} "
                            f"({scheduler_event.start_time})",
                            "tid": placement.task_graph,
                            "args": {
                                "name": placement.task_name,
                                "task_graph": placement.task_graph,
                                "id": str(placement.id),
                                "placement_time": placement.placement_time,
                                "timestamp": placement.timestamp,
                                "worker_pool": placement.worker_pool,
                                "intended_completion_time": placement.completion_time,
                                "deadline": placement.deadline,
                            },
                        }
                        trace["traceEvents"].append(trace_event)

        if trace_fmt == "resource":
            resource_ids_to_canonical_names = {}
            for worker_pool in self.get_worker_pools(csv_path):
                resource_counter = defaultdict(int)
                for resource in worker_pool.resources:
                    resource_counter[resource.name] += 1
                    resource_ids_to_canonical_names[resource.id] = (
                        f"{resource.name}_{resource_counter[resource.name]}"
                    )

        # Output all the tasks and the requested deadlines.
        if trace_fmt in ["task", "taskgraph", "application", "resource"]:
            for task in self.get_tasks(csv_path):
                # Do not output the tasks if it does not fall within the given time.
                if not check_if_time_intersects(
                    between_time, task.start_time, task.completion_time
                ):
                    continue
                if trace_fmt in ["task", "taskgraph", "application"]:
                    if trace_fmt == "task":
                        if "." in task.name:
                            # pid = operator name, tid = callback name
                            pid, tid = task.name.split(".", 1)
                        else:
                            pid = tid = task.name
                    else:
                        if trace_fmt == "taskgraph":
                            pid = task.task_graph
                            tid = task.name
                        else:
                            pid = task.task_graph.split("@", 1)[0]
                            tid = task.task_graph

                    # Output the task's placement as individual elements.
                    for placement in task.placements:
                        if placement.completion_time is None:
                            # In case the task didn't finish in the simulation.
                            continue
                        trace_event = {
                            "name": f"{task.name}::{task.timestamp}",
                            "cat": "task,duration",
                            "ph": "X",
                            "ts": placement.placement_time,
                            "dur": placement.completion_time - placement.placement_time,
                            "pid": pid,
                            "tid": tid,
                            "args": {
                                "name": task.name,
                                "task_graph": task.task_graph,
                                "id": str(task.task_id),
                                "timestamp": task.timestamp,
                                "intended_release_time": task.intended_release_time,
                                "release_time": task.release_time,
                                "runtime": task.runtime,
                                "deadline": task.deadline,
                                "start_time": task.start_time,
                                "completion_time": task.completion_time,
                                "missed_deadline": task.missed_deadline,
                                "skipped_times": task.skipped_times,
                            },
                        }
                        trace["traceEvents"].append(trace_event)

                    # Output the release times.
                    if (
                        show_release_times == "intended"
                        and task.intended_release_time != -1
                    ) or show_release_times == "always":
                        # The scope of the release time events is per thread if we
                        # are outputting a task focused trace, and process if we are
                        # outputting an application or a taskgraph focused trace.
                        scope = "t" if trace_fmt == "task" else "p"
                        timestamp = (
                            task.intended_release_time
                            if show_release_times == "intended"
                            else task.release_time
                        )
                        trace_event = {
                            "name": f"{task.name}::{task.timestamp}",
                            "cat": "task,releasetime,intended,instant",
                            "ph": "i",
                            "ts": timestamp,
                            "pid": pid,
                            "tid": tid,
                            "s": scope,
                        }
                        trace["traceEvents"].append(trace_event)

                    # Output the deadline.
                    if (
                        show_deadlines == "missed" and task.missed_deadline
                    ) or show_deadlines == "always":
                        # The scope of the missed deadline events is per thread if we
                        # are outputting a task focused trace, and process if we are
                        # outputting an application or a taskgraph focused trace.
                        scope = "t" if trace_fmt == "task" else "p"
                        trace_event = {
                            "name": f"{task.name}::{task.timestamp}",
                            "cat": "task,missed,deadline,instant",
                            "ph": "i",
                            "ts": task.deadline,
                            "pid": pid,
                            "tid": tid,
                            "s": scope,
                        }
                        trace["traceEvents"].append(trace_event)

                    # Output the release time.
                elif trace_fmt == "resource":
                    # Output the task's placement as individual elements.
                    for placement in task.placements:
                        if placement.completion_time is None:
                            # The task didn't finish by the end of the simulation.
                            continue
                        tids = [
                            resource_ids_to_canonical_names[resource.id]
                            for resource in placement.resources_used
                        ]
                        for tid in tids:
                            trace_event = {
                                "name": f"{task.name}::{task.timestamp}",
                                "cat": "task,duration",
                                "ph": "X",
                                "ts": placement.placement_time,
                                "dur": placement.completion_time
                                - placement.placement_time,
                                "pid": placement.worker_pool.name,
                                "tid": tid,
                                "args": {
                                    "name": task.name,
                                    "task_graph": task.task_graph,
                                    "id": str(task.task_id),
                                    "timestamp": task.timestamp,
                                    "intended_release_time": task.intended_release_time,
                                    "release_time": task.release_time,
                                    "runtime": task.runtime,
                                    "deadline": task.deadline,
                                    "start_time": task.start_time,
                                    "completion_time": task.completion_time,
                                    "missed_deadline": task.missed_deadline,
                                    "skipped_times": task.skipped_times,
                                    "worker_name": placement.worker_pool.name,
                                },
                            }
                            trace["traceEvents"].append(trace_event)

                            # Output the deadline.
                            if (
                                show_deadlines == "missed" and task.missed_deadline
                            ) or show_deadlines == "always":
                                if check_if_time_intersects(
                                    task.deadline,
                                    placement.placement_time,
                                    placement.completion_time,
                                ):
                                    trace_event = {
                                        "name": f"{task.name}::{task.timestamp}",
                                        "cat": "task,missed,deadline,instant",
                                        "ph": "i",
                                        "ts": task.deadline,
                                        "pid": placement.worker_pool.name,
                                        "tid": tid,
                                        # The scope of missed deadline events is thread.
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
                    between_time, task.release_time, task.release_time + task.runtime
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
                            "intended_release_time": task.intended_release_time,
                            "release_time": task.release_time,
                            "runtime": task.runtime,
                            "deadline": task.deadline,
                            "start_time": task.start_time,
                            "completion_time": task.completion_time,
                            "missed_deadline": task.missed_deadline,
                            "skipped_times": task.skipped_times,
                        },
                    }
                    trace["traceEvents"].append(trace_event)

                    # Output the deadline event according to the required strategy.
                    if (
                        show_deadlines == "missed" and task.missed_deadline
                    ) or show_deadlines == "always":
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
            json.dump(trace, f, default=lambda obj: obj.__dict__, indent=4)
