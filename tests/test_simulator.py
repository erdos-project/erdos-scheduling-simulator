from typing import Sequence

import pytest

from schedulers import BaseScheduler
from simulator import Event, EventQueue, EventType, Simulator
from tests.utils import create_default_task
from workers import Worker, WorkerPool
from workload import Job, Resource, Resources, TaskGraph, TaskState


class MockScheduler(BaseScheduler):
    """ A MockScheduler that enables the testing of the Simulator. """

    def __init__(self, runtime: int, placement):
        self._runtime = runtime
        self._task_placement = placement
        self.preemptive = False
        self.scheduling_horizon = 0

    def schedule(self,
                 sim_time=None,
                 released_tasks=None,
                 task_graph=None,
                 worker_pools=None):
        return (self._runtime, self._task_placement)


def __create_default_worker_pool(
    resources: Sequence[Resources] = [
        Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1,
        }),
        Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1,
        }),
    ]
) -> WorkerPool:
    """ Creates an instance of WorkerPool with a given set of resources. """
    workers = []
    for index, _resources in enumerate(resources):
        workers.append(Worker(
            name=f"Worker_{index}",
            resources=_resources,
        ))
    return WorkerPool(name="WorkerPool_Test", workers=workers)


def test_event_construction_success():
    """ Test that an Event can be successfully constructed. """
    event = Event(event_type=EventType.SCHEDULER_START, time=1)
    assert event.event_type == EventType.SCHEDULER_START, "Invalid Event type."
    assert event.time == 1, "Invalid event time."


def test_event_construction_failure():
    """ Test that an Event without a task fails to be constructed. """
    with pytest.raises(ValueError):
        Event(event_type=EventType.TASK_RELEASE, time=1)


def test_event_queue_one():
    """ Test that an EventQueue properly orders the given events. """
    event_queue = EventQueue()
    event_queue.add_event(Event(event_type=EventType.SCHEDULER_START, time=1))
    event_queue.add_event(
        Event(event_type=EventType.SCHEDULER_FINISHED, time=2))

    assert len(event_queue) == 2, "Incorrect number of events in the queue."
    assert event_queue.next().event_type == EventType.SCHEDULER_START,\
        "Incorrect event returned by the queue."

    assert len(event_queue) == 1, "Incorrect number of events in the queue."
    assert event_queue.next().event_type == EventType.SCHEDULER_FINISHED,\
        "Incorrect event returned by the queue."


def test_event_queue_two():
    """ Test that an EventQueue properly orders the given events. """
    event_queue = EventQueue()
    event_queue.add_event(
        Event(event_type=EventType.TASK_RELEASE,
              time=3,
              task=create_default_task()))
    event_queue.add_event(
        Event(event_type=EventType.SCHEDULER_FINISHED, time=2))
    event_queue.add_event(Event(event_type=EventType.SCHEDULER_START, time=1))

    assert len(event_queue) == 3, "Incorrect number of events in the queue."
    assert event_queue.next().event_type == EventType.SCHEDULER_START,\
        "Incorrect event returned by the queue."

    assert len(event_queue) == 2, "Incorrect number of events in the queue."
    assert event_queue.next().event_type == EventType.SCHEDULER_FINISHED,\
        "Incorrect event returned by the queue."

    assert len(event_queue) == 1, "Incorrect number of events in the queue."
    assert event_queue.next().event_type == EventType.TASK_RELEASE,\
        "Incorrect event returned by the queue."


def test_simulator_construction():
    """ Test that a simulator can be correctly constructed. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=1, placement=None),
                          job_graph=None)
    assert len(simulator._worker_pools) == 1, "Incorrect number of WorkerPool"
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events in the EventQueue."


def test_failed_construction_of_scheduler_start_event():
    """ Test if a non SCHEDULER_FINISHED event raises an error or not. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=1, placement=None),
                          job_graph=None)
    with pytest.raises(ValueError):
        simulator._Simulator__get_next_scheduler_event(
            event=Event(event_type=EventType.SCHEDULER_START, time=3),
            task_graph=TaskGraph(),
            scheduler_frequency=-1,
            last_scheduler_start_time=2)


def test_construction_of_scheduler_start_event():
    """ Test the correct construction of a SCHEDULER_START event. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=1, placement=None),
                          job_graph=None)

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(event_type=EventType.SCHEDULER_FINISHED, time=3),
        task_graph=TaskGraph(),
        scheduler_frequency=-1,
        last_scheduler_start_time=1)
    assert simulator_start_event.time == 4,\
        "Incorrect start time for Scheduler."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(event_type=EventType.SCHEDULER_FINISHED, time=3),
        task_graph=TaskGraph(),
        scheduler_frequency=5,
        last_scheduler_start_time=1)
    assert simulator_start_event.time == 6,\
        "Incorrect start time for Scheduler."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(event_type=EventType.SCHEDULER_FINISHED, time=7),
        task_graph=TaskGraph(),
        scheduler_frequency=5,
        last_scheduler_start_time=1)
    assert simulator_start_event.time == 8,\
        "Incorrect start time for Scheduler."


def test_simulator_loop_finish_event():
    """ Test that a correct SIMULATOR_END event is generated. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=1, placement=None),
                          job_graph=None)
    assert simulator._event_queue.next().event_type ==\
        EventType.SIMULATOR_START, "Incorrect event received."
    assert simulator._event_queue.next().event_type ==\
        EventType.SCHEDULER_START, "Incorrect event received."
    assert len(simulator._event_queue) == 0,\
        "Incorrect length for event queue."

    simulator_start_event = simulator._Simulator__get_next_scheduler_event(
        event=Event(event_type=EventType.SCHEDULER_FINISHED, time=3),
        task_graph=TaskGraph(),
        scheduler_frequency=-1,
        last_scheduler_start_time=1)
    assert simulator_start_event.event_type == EventType.SIMULATOR_END,\
        "Incorrect event received from next_scheduler_event"


def test_scheduler_invocation_by_simulator():
    """ Test that the simulator correctly invokes the scheduler. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=5, placement=None),
                          job_graph=None)
    scheduler_finished_event = simulator._Simulator__run_scheduler(
        event=Event(event_type=EventType.SCHEDULER_START, time=1),
        task_graph=TaskGraph(),
    )
    assert scheduler_finished_event.time == 6, "Incorrect finish time."


def test_simulator_step():
    """ Test that the Simulator's step() method correctly inserts events. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=5, placement=None),
                          job_graph=None)

    # Create, release and place a Task.
    task = create_default_task(runtime=3)
    task.release(1)
    worker_pool.place_task(task)
    task.start(2)

    assert simulator._simulator_time == 0, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events."

    # Step through the execution.
    simulator._Simulator__step()
    assert simulator._simulator_time == 1, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == 2, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == 3, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == 4, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 2,\
        "Incorrect number of starting events."
    simulator._Simulator__step()
    assert simulator._simulator_time == 5, "Incorrect starting simulator time."
    assert len(simulator._event_queue) == 3,\
        "Incorrect number of starting events."

    # Check the order of events in the queue.
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.SIMULATOR_START,\
        "Incorrect event type."
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.SCHEDULER_START,\
        "Incorrect event type."
    next_event = simulator._event_queue.next()
    assert next_event.event_type == EventType.TASK_FINISHED,\
        "Incorrect event type."


def test_simulator_handle_event():
    """ Test the Simulator's handle_event method with different events. """
    worker_pool = __create_default_worker_pool()
    simulator = Simulator(worker_pools=[worker_pool],
                          scheduler=MockScheduler(runtime=5, placement=None),
                          job_graph=None)

    # Test the SIMULATOR_START event.
    return_value = simulator._Simulator__handle_event(event=Event(
        event_type=EventType.SIMULATOR_START, time=1),
                                                      task_graph=TaskGraph())
    assert not return_value, "Incorrect return value for event type."

    # Test the SIMULATOR_END event.
    return_value = simulator._Simulator__handle_event(event=Event(
        event_type=EventType.SIMULATOR_END, time=1),
                                                      task_graph=TaskGraph())
    assert return_value, "Incorrect return value for event type."

    # Test the TASK_FINISHED event.
    perception_task = create_default_task(job=Job(name="Perception"))
    planning_task = create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    task_graph.add_task(perception_task, [planning_task])
    perception_task.release(2)
    perception_task.start(3)
    perception_task.update_remaining_time(0)
    perception_task.finish(4)

    assert len(simulator._event_queue) == 2, "Incorrect length of EventQueue."
    return_value = simulator._Simulator__handle_event(event=Event(
        event_type=EventType.TASK_FINISHED, time=4, task=perception_task),
                                                      task_graph=task_graph)
    assert not return_value, "Incorrect return value for event type."
    assert len(simulator._event_queue) == 3, "Incorrect length of EventQueue."

    # Test the SCHEDULER_START event.
    return_value = simulator._Simulator__handle_event(event=Event(
        event_type=EventType.SCHEDULER_START,
        time=5,
    ),
                                                      task_graph=TaskGraph())
    assert not return_value, "Incorrect return value for event type."
    assert len(simulator._event_queue) == 4, "Incorrect length of EventQueue."

    # Test the SCHEDULER_FINISHED event.
    simulator._last_task_placement = [(planning_task, worker_pool.id)]
    return_value = simulator._Simulator__handle_event(event=Event(
        event_type=EventType.SCHEDULER_FINISHED,
        time=6,
    ),
                                                      task_graph=TaskGraph())
    assert len(simulator._event_queue) == 5, "Incorrect length of EventQueue."
    assert planning_task.state == TaskState.RUNNING, "Wrong task state."
    assert len(worker_pool.get_placed_tasks()) == 1,\
        "Incorrect number of placed tasks on WorkerPool."
    assert worker_pool.get_placed_tasks()[0] == planning_task,\
        "Incorrect task placed on the WorkerPool."
