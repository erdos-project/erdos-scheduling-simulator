from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence


@dataclass
class WorkerPoolStats:
    simulator_time: int
    resource_utilizations: dict[str, tuple[float, float]]


@dataclass
class Resource:
    name: str
    id: str
    quantity: float

    def __post_init__(self):
        self.quantity = float(self.quantity)

    def __repr__(self) -> str:
        return f"Resource(name={self.name}, id={self.id}, quantity={self.quantity})"


@dataclass
class WorkerPoolUtilization:
    simulator_time: int
    resource_name: str
    allocated_quantity: float
    available_quantity: float


@dataclass
class WorkerPool:
    name: str
    id: str
    resources: Sequence[Resource] = field(default_factory=list)
    utilizations: Sequence[WorkerPoolUtilization] = field(default_factory=list)


@dataclass
class Placement:
    task_name: str
    timestamp: int
    task_id: str
    task_graph: str
    placement_time: int
    deadline: int
    worker_pool: WorkerPool
    resources_used: Sequence[Resource] = field(default_factory=list)
    completion_time: Optional[int] = None


@dataclass
# @total_ordering
class Task:
    name: str
    task_graph: str
    timestamp: int
    task_id: str
    intended_release_time: Optional[int]
    release_time: Optional[int]
    deadline: Optional[int]
    window_to_execute: Optional[int]
    # The runtime of the slowest execution strategy
    slowest_execution_time: Optional[int] = None

    # The actual task runtime when it get placed
    runtime: Optional[int] = None
    placements: list[Placement] = field(init=False, default_factory=list)
    start_time: Optional[int] = None
    placement_time: Optional[int] = None
    skipped_times: List = field(init=False, default_factory=list)
    completion_time: Optional[int] = None
    slack: Optional[int] = None
    missed_deadline: bool = False
    deadline_miss_detected_at: Optional[int] = None
    cancelled: bool = False
    cancelled_at: Optional[int] = None

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

    def update_placement(
        self, csv_reading: str, worker_pools: Mapping[str, WorkerPool]
    ):
        """Updates the values of the Task based on the TASK_PLACEMENT event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_PLACEMENT`.
            worker_pools (Mapping[str, WorkerPool]): A name to WorkerPool mapping to
                allow tasks to directly reference WorkerPools.
        """
        assert (
            csv_reading[1] == "TASK_PLACEMENT"
        ), f"The event {csv_reading[1]} was not of type TASK_PLACEMENT."

        placement_time = int(csv_reading[0])
        placement = Placement(
            task_name=self.name,
            timestamp=self.timestamp,
            task_id=self.task_id,
            task_graph=self.task_graph,
            deadline=self.deadline,
            placement_time=placement_time,
            worker_pool=worker_pools[csv_reading[6]],
            resources_used=[
                Resource(*csv_reading[i : i + 3]) for i in range(8, len(csv_reading), 3)
            ],
        )
        self.placements.append(placement)
        if not self.start_time or self.start_time > placement_time:
            self.start_time = placement_time
        if not self.placement_time or self.placement_time > placement_time:
            self.placement_time = placement_time
        self.runtime = int(csv_reading[7])

    def update_skip(self, csv_reading: str):
        """Updates the values of the Task based on the TASK_SKIP event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_SKIP`.
        """
        assert (
            csv_reading[1] == "TASK_SKIP"
        ), f"The event {csv_reading[1]} was not of type TASK_SKIP."
        self.skipped_times.append(int(csv_reading[0]))

    def update_preempt(self, csv_reading: str):
        """Updates the placement information of the Task based on the TASK_PREEMPT
        event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_PREEMPT`.
        """
        assert (
            csv_reading[1] == "TASK_PREEMPT"
        ), f"The event {csv_reading[1]} was not of type TASK_PREEMPT."
        self.placements[-1].completion_time = int(csv_reading[0])

    def update_migration(
        self, csv_reading: str, worker_pools: Mapping[str, WorkerPool]
    ):
        """Updates the placement information of the Task based on the TASK_MIGRATED
        event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_MIGRATED`.
            worker_pools (Mapping[str, WorkerPool]): A name to WorkerPool mapping to
                allow tasks to directly reference WorkerPools.
        """
        assert (
            csv_reading[1] == "TASK_MIGRATED"
        ), f"The event {csv_reading[1]} was not of type TASK_MIGRATED."
        placement = Placement(
            name=self.name,
            timestamp=self.timestamp,
            task_id=self.id,
            task_graph=self.task_graph,
            placement_time=int(csv_reading[0]),
            deadline=self.deadline,
            worker_pool=worker_pools[csv_reading[5]],
            resources_used=[
                Resource(*csv_reading[i : i + 3]) for i in range(7, len(csv_reading), 3)
            ],
        )
        self.placements.append(placement)
        if not self.start_time or self.start_time > placement.placement_time:
            self.start_time = placement.placement_time
        if not self.placement_time or self.placement_time > placement.placement_time:
            self.placement_time = placement.placement_time

    def update_finish(self, csv_reading: str):
        """Updates the values of the Task based on the TASK_FINISHED event from CSV.

        Args:
            csv_reading (str): The CSV reading of type `TASK_FINISHED`.
        """
        assert (
            csv_reading[1] == "TASK_FINISHED"
        ), f"The event {csv_reading[1]} was not of type TASK_FINISHED."
        self.completion_time = int(csv_reading[5])
        self.slack = self.deadline - self.completion_time
        self.placements[-1].completion_time = int(csv_reading[5])

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

    @property
    def was_placed(self):
        return len(self.placements) > 0

    @property
    def was_completed(self):
        return self.completion_time is not None

    @property
    def release_time_compare_key(self):
        return self.release_time if self.release_time is not None else float("inf")

    def __str__(self):
        return f"Task(name={self.name}, timestamp={self.timestamp})"

    def __repr__(self):
        return str(self)

    def __lt__(self, other: "Task"):
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


@dataclass
class TaskGraph(object):
    name: str
    release_time: int
    deadline: int
    num_tasks: int
    critical_path_time: Optional[int] = None
    window_to_execute: int = field(init=False)

    cancelled: bool = False
    cancelled_at: Optional[int] = None

    # Values updated from the TASK_GRAPH_FINISHED event.
    completion_at: Optional[int] = None
    slack: Optional[int] = None
    deadline_miss_detected_at: Optional[int] = None

    def __post_init__(self):
        self.window_to_execute: int = self.deadline - self.release_time

    @property
    def was_completed(self):
        return self.completion_at is not None

    @property
    def missed_deadline(self):
        return self.completion_at is not None and self.completion_at > self.deadline

    def __str__(self) -> str:
        return (
            f"TaskGraph(name={self.name}, num_tasks={self.num_tasks},"
            + f"release_time={self.release_time}, deadline={self.deadline}),"
            + f"critical_path_time={self.critical_path_time}, slack={self.slack},"
            + f"cancelled={self.cancelled}, cancelled_at={self.cancelled_at},"
            + f"completion_at={self.completion_at},"
            + f"deadline_miss_detected_at={self.deadline_miss_detected_at})"
        )


class Scheduler(object):
    def __init__(
        self,
        start_time: int,
        released_tasks: int,
        previously_placed_tasks: int,
        instance_id: int,
    ):
        self.start_time = start_time
        self.released_tasks = released_tasks
        self.previously_placed_tasks = previously_placed_tasks
        self.total_tasks = released_tasks + previously_placed_tasks
        self.instance_id = instance_id

        # Values updated with the SCHEDULER_FINISHED event.
        self.end_time = None
        self.runtime = None
        self.true_runtime = None
        self.num_placed_tasks = None
        self.num_unplaced_tasks = None

        # Values updated from TASK_SCHEDULED events.
        self.task_placements = []

    def update_finish(self, csv_reading: str):
        """Updates the values of the Scheduler based on the SCHEDULER_FINISHED event
        from CSV.

        Args:
            csv_reading (str): The CSV reading of type `SCHEDULER_FINISHED`.
        """
        assert (
            csv_reading[1] == "SCHEDULER_FINISHED"
        ), f"The event {csv_reading[1]} was not of type SCHEDULER_FINISHED."
        assert (
            self.end_time is None
        ), f"The Scheduler at {self.start_time} was already finished."
        self.end_time = int(csv_reading[0])
        self.runtime = int(csv_reading[2])
        self.num_placed_tasks = int(csv_reading[3])
        self.num_unplaced_tasks = int(csv_reading[4])
        self.true_runtime = int(csv_reading[5])

    def update_task_schedule(self, csv_reading: List[str]):
        assert (
            csv_reading[1] == "TASK_SCHEDULED"
        ), f"Event {csv_reading[1]} is not of type TASK_SCHEDULED."
        self.task_placements.append(
            Placement(
                task_name=csv_reading[2],
                timestamp=int(csv_reading[4]),
                task_id=csv_reading[5],
                task_graph=csv_reading[3],
                placement_time=int(csv_reading[7]),
                deadline=csv_reading[6],
                worker_pool=csv_reading[8],
                completion_time=int(csv_reading[7]) + int(csv_reading[9]),
            )
        )


class Simulator(object):
    def __init__(self, csv_path: str, start_time: int, total_tasks: int = 0):
        self.csv_path = csv_path
        self.start_time = start_time
        self.total_tasks = total_tasks

        # Values updated from the SIMULATOR_END event.
        self.end_time = None
        self.finished_tasks = None
        self.dropped_tasks = None
        self.missed_deadlines = None
        self.finished_task_graphs = None
        self.dropped_taskgraphs = None
        self.missed_taskgraphs = None
        self.goodput_taskgraphs = None

        self.worker_pools = []
        self.tasks: list[WorkerPool] = []
        self.scheduler_invocations: list[Scheduler] = []
        self.task_graphs: dict[str, TaskGraph] = {}

    def update_stats(self, csv_reading: str):
        assert (
            csv_reading[1] == "LOG_STATS"
        ), f"The event {csv_reading[1]} was not of type LOG_STATS."
        self.finished_tasks = int(csv_reading[2])
        self.dropped_tasks = int(csv_reading[3])
        self.missed_deadlines = int(csv_reading[4])
        self.finished_task_graphs = int(csv_reading[5])
        self.dropped_taskgraphs = int(csv_reading[6])
        self.missed_taskgraphs = int(csv_reading[7])
        self.goodput_taskgraphs = self.finished_task_graphs - self.missed_taskgraphs

    def update_finish(self, csv_reading: str):
        """Updates the values of the Simulator based on the SIMULATOR_END event from
        CSV.

        Args:
            csv_reading (str): The CSV reading of type `SIMULATOR_END`.
        """
        assert (
            csv_reading[1] == "SIMULATOR_END"
        ), f"The event {csv_reading[1]} was not of type SIMULATOR_END."
        self.end_time = int(csv_reading[0])
        # self.finished_tasks = int(csv_reading[2])
        # self.dropped_tasks = int(csv_reading[3])
        # self.missed_deadlines = int(csv_reading[4])
        # self.finished_task_graphs = int(csv_reading[5])
        # self.dropped_taskgraphs = int(csv_reading[6])
        # self.missed_taskgraphs = int(csv_reading[7])
        # self.goodput_taskgraphs = self.finished_task_graphs - self.missed_taskgraphs
