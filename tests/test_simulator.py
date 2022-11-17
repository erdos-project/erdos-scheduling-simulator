from typing import Sequence

import pytest

from schedulers import BaseScheduler
from simulator import Event, EventQueue, EventType, Simulator
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool
from workload import (
    Job,
    Placement,
    Placements,
    Resource,
    Resources,
    TaskGraph,
    TaskState,
    Workload,
)
from workload.workload import Workload


class MockScheduler(BaseScheduler):
    """A MockScheduler that enables the testing of the Simulator."""

    def __init__(self, runtime: EventTime, placement=[]):
        self._runtime = runtime
        self._task_placement = placement
        self._preemptive = False
        self._lookahead = EventTime(0, EventTime.Unit.US)

    def schedule(
        self, sim_time=None, released_tasks=None, task_graph=None, worker_pools=None
    ) -> Placements:
        return Placements(self._runtime, self._task_placement)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def lookahead(self):
        return self._lookahead


def __create_default_worker_pool(
    resources: Sequence[Resources] = [
        Resources(
            {
                Resource(name="CPU"): 1,
                Resource(name="GPU"): 1,
            }
        ),
        Resources(
            {
                Resource(name="CPU"): 1,
                Resource(name="GPU"): 1,
            }
        ),
    ]
) -> WorkerPool:
    """Creates an instance of WorkerPool with a given set of resources."""
    workers = []
    for index, _resources in enumerate(resources):
        workers.append(
            Worker(
                name=f"Worker_{index}",
                resources=_resources,
            )
        )
    return WorkerPool(name="WorkerPool_Test", workers=workers)


def test_event_construction_success():
    """Test that an Event can be successfully constructed."""
    event_time = EventTime(1, EventTime.Unit.US)
    event = Event(event_type=EventType.SCHEDULER_START, time=event_time)
    assert event.event_type == EventType.SCHEDULER_START, "Invalid Event type."
    assert event.time == event_time, "Invalid event time."


def test_event_construction_failure():
    """Test that an Event without a task fails to be constructed."""
    event_time = EventTime(1, EventTime.Unit.US)
    with pytest.raises(ValueError):
        Event(event_type=EventType.TASK_RELEASE, time=event_time)


def test_event_queue_one():
    """Test that an EventQueue properly orders the given events."""
    event_queue = EventQueue()
    event_queue.add_event(
        Event(
            event_type=EventType.SCHEDULER_START, time=EventTime(1, EventTime.Unit.US)
        )
    )
    event_queue.add_event(
        Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(2, EventTime.Unit.US),
        )
    )

    assert len(event_queue) == 2, "Incorrect number of events in the queue."
    assert (
        event_queue.next().event_type == EventType.SCHEDULER_START
    ), "Incorrect event returned by the queue."

    assert len(event_queue) == 1, "Incorrect number of events in the queue."
    assert (
        event_queue.next().event_type == EventType.SCHEDULER_FINISHED
    ), "Incorrect event returned by the queue."


def test_event_queue_two():
    """Test that an EventQueue properly orders the given events."""
    event_queue = EventQueue()
    event_queue.add_event(
        Event(
            event_type=EventType.TASK_RELEASE,
            time=EventTime(3, EventTime.Unit.US),
            task=create_default_task(),
        )
    )
    event_queue.add_event(
        Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(2, EventTime.Unit.US),
        )
    )
    event_queue.add_event(
        Event(
            event_type=EventType.SCHEDULER_START, time=EventTime(1, EventTime.Unit.US)
        )
    )

    assert len(event_queue) == 3, "Incorrect number of events in the queue."
    assert (
        event_queue.next().event_type == EventType.SCHEDULER_START
    ), "Incorrect event returned by the queue."

    assert len(event_queue) == 2, "Incorrect number of events in the queue."
    assert (
        event_queue.next().event_type == EventType.SCHEDULER_FINISHED
    ), "Incorrect event returned by the queue."

    assert len(event_queue) == 1, "Incorrect number of events in the queue."
    assert (
        event_queue.next().event_type == EventType.TASK_RELEASE
    ), "Incorrect event returned by the queue."


def test_simulator_construction():
    """Test that a simulator can be correctly constructed."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(1, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )
    assert len(simulator._worker_pools) == 1, "Incorrect number of WorkerPool"
    assert (
        len(simulator._event_queue) == 2
    ), "Incorrect number of starting events in the EventQueue."


def test_failed_construction_of_scheduler_start_event():
    """Test if a non SCHEDULER_FINISHED event raises an error or not."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(1, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )
    with pytest.raises(ValueError):
        simulator._Simulator__get_next_scheduler_event(
            event=Event(
                event_type=EventType.SCHEDULER_START,
                time=EventTime(3, EventTime.Unit.US),
            ),
            scheduler_frequency=EventTime(-1, EventTime.Unit.US),
            last_scheduler_start_time=EventTime(2, EventTime.Unit.US),
        )


def test_construction_of_scheduler_start_event():
    """Test the correct construction of a SCHEDULER_START event."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(1, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(3, EventTime.Unit.US),
        ),
        scheduler_frequency=EventTime(-1, EventTime.Unit.US),
        last_scheduler_start_time=EventTime(1, EventTime.Unit.US),
    )
    assert simulator_start_event.time == EventTime(
        4, EventTime.Unit.US
    ), "Incorrect start time for Scheduler."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(3, EventTime.Unit.US),
        ),
        scheduler_frequency=EventTime(5, EventTime.Unit.US),
        last_scheduler_start_time=EventTime(1, EventTime.Unit.US),
    )
    assert simulator_start_event.time == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect start time for Scheduler."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(7, EventTime.Unit.US),
        ),
        scheduler_frequency=EventTime(5, EventTime.Unit.US),
        last_scheduler_start_time=EventTime(1, EventTime.Unit.US),
    )
    assert simulator_start_event.time == EventTime(
        8, EventTime.Unit.US
    ), "Incorrect start time for Scheduler."


def test_simulator_loop_finish_event():
    """Test that a correct SIMULATOR_END event is generated."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(1, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )
    assert (
        simulator._event_queue.next().event_type == EventType.SIMULATOR_START
    ), "Incorrect event received."
    assert (
        simulator._event_queue.next().event_type == EventType.SCHEDULER_START
    ), "Incorrect event received."
    assert len(simulator._event_queue) == 0, "Incorrect length for event queue."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(3, EventTime.Unit.US),
        ),
        scheduler_frequency=EventTime(-1, EventTime.Unit.US),
        last_scheduler_start_time=EventTime(1, EventTime.Unit.US),
    )
    assert (
        simulator_start_event.event_type == EventType.SIMULATOR_END
    ), "Incorrect event received from next_scheduler_event"


def test_scheduler_invocation_by_simulator():
    """Test that the simulator correctly invokes the scheduler."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(5, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )
    scheduler_finished_event = simulator._Simulator__run_scheduler(
        event=Event(
            event_type=EventType.SCHEDULER_START, time=EventTime(1, EventTime.Unit.US)
        ),
    )
    assert scheduler_finished_event.time == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect finish time."


def test_simulator_step():
    """Test that the Simulator's step() method correctly inserts events."""
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(5, EventTime.Unit.US), placement=[]),
        workload=Workload.empty(),
    )

    # Create, release and place a Task.
    task = create_default_task(runtime=3)
    task.release(EventTime(1, EventTime.Unit.US))
    task.schedule(EventTime(2, EventTime.Unit.US), worker_pool.id)
    worker_pool.place_task(task)
    task.start(EventTime(2, EventTime.Unit.US))

    assert simulator._simulator_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2, "Incorrect number of starting events."

    # Step through the execution.
    simulator._Simulator__step()
    assert simulator._simulator_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2, "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == EventTime(
        2, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2, "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == EventTime(
        3, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2, "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == EventTime(
        4, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2, "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == EventTime(
        5, EventTime.Unit.US
    ), "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 3, "Incorrect number of starting events."

    # Check the order of events in the queue.
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.SIMULATOR_START, "Incorrect event type."
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.SCHEDULER_START, "Incorrect event type."
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.TASK_FINISHED, "Incorrect event type."


def test_simulator_handle_event():
    """Test the Simulator's handle_event method with different events."""
    worker_pool = __create_default_worker_pool()
    perception_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    planning_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph()
    task_graph.add_task(perception_task, [planning_task])
    simulator = Simulator(
        worker_pools=[worker_pool],
        scheduler=MockScheduler(runtime=EventTime(5, EventTime.Unit.US), placement=[]),
        workload=Workload.from_task_graphs({perception_task.task_graph: task_graph}),
    )

    # Test the SIMULATOR_START event.
    return_value = simulator._Simulator__handle_event(
        event=Event(
            event_type=EventType.SIMULATOR_START, time=EventTime(1, EventTime.Unit.US)
        ),
    )
    assert not return_value, "Incorrect return value for event type."

    # Test the SIMULATOR_END event.
    return_value = simulator._Simulator__handle_event(
        event=Event(
            event_type=EventType.SIMULATOR_END, time=EventTime(1, EventTime.Unit.US)
        ),
    )
    assert return_value, "Incorrect return value for event type."

    # Test the TASK_FINISHED event.
    perception_task.release(EventTime(2, EventTime.Unit.US))
    perception_task.schedule(EventTime(2, EventTime.Unit.US))
    perception_task.start(EventTime(3, EventTime.Unit.US))
    perception_task.update_remaining_time(EventTime(0, EventTime.Unit.US))
    perception_task.finish(EventTime(4, EventTime.Unit.US))

    assert len(simulator._event_queue) == 2, "Incorrect length of EventQueue."
    return_value = simulator._Simulator__handle_event(
        event=Event(
            event_type=EventType.TASK_FINISHED,
            time=EventTime(4, EventTime.Unit.US),
            task=perception_task,
        ),
    )
    assert not return_value, "Incorrect return value for event type."
    assert len(simulator._event_queue) == 3, "Incorrect length of EventQueue."

    # Test the SCHEDULER_START event.
    return_value = simulator._Simulator__handle_event(
        event=Event(
            event_type=EventType.SCHEDULER_START,
            time=EventTime(5, EventTime.Unit.US),
        ),
    )
    assert not return_value, "Incorrect return value for event type."
    assert len(simulator._event_queue) == 4, "Incorrect length of EventQueue."

    # Test the SCHEDULER_FINISHED event.
    simulator._last_task_placement = [
        Placement(planning_task, worker_pool.id, EventTime(6, EventTime.Unit.US))
    ]
    return_value = simulator._Simulator__handle_event(
        event=Event(
            event_type=EventType.SCHEDULER_FINISHED,
            time=EventTime(6, EventTime.Unit.US),
        ),
    )
    assert len(simulator._event_queue) == 7, "Incorrect length of EventQueue."
