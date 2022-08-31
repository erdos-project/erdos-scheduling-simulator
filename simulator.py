import heapq
import sys
from enum import Enum
from functools import total_ordering
from operator import attrgetter
from typing import Optional, Sequence, Type

import absl  # noqa: F401

from schedulers import BaseScheduler
from utils import EventTime, setup_csv_logging, setup_logging
from workers import WorkerPool, WorkerPools
from workload import JobGraph, Resource, Resources, Task, TaskGraph, TaskState
from workload.workload import Workload


@total_ordering
class EventType(Enum):
    """Represents the different Events that a simulator has to simulate."""

    SIMULATOR_START = 0  # Signify the start of the simulator loop.
    TASK_RELEASE = 1  # Ask the simulator to release the task.
    TASK_FINISHED = 2  # Notify the simulator of the end of a task.
    TASK_PREEMPT = 3  # Ask the simulator to preempt a task.
    TASK_MIGRATION = 4  # Ask the simulator to migrate a task.
    TASK_PLACEMENT = 5  # Ask the simulator to place a task.
    SCHEDULER_START = 6  # Requires the simulator to invoke the scheduler.
    SCHEDULER_FINISHED = 7  # Signifies the end of the scheduler loop.
    SIMULATOR_END = 8  # Signify the end of the simulator loop.
    LOG_UTILIZATION = 9  # Ask the simulator to log the utilization of the worker pools.

    def __lt__(self, other):
        # This method is used to order events in the event queue. We prioritize
        # events that first free resources.
        return self.value < other.value


class Event(object):
    """Represents an `Event` that is managed by the `EventQueue` and informs
    what action the simulator needs to do at the given simulator time.

    Args:
        event_type (`EventType`): The type of the event.
        time (`EventTime`): The simulator time at which the event occurred.
        task (`Optional[Task]`): The task associated with this event if it is
            of type `TASK_RELEASE` or `TASK_FINISHED`.

    Raises:
        `ValueError` if the event is of type `TASK_RELEASE` or `TASK_FINISHED`
        and no associated task is provided, or if the time is not of type `EventTime`.
    """

    def __init__(
        self,
        event_type: EventType,
        time: EventTime,
        task: Optional[Task] = None,
        placement: Optional[str] = None,
    ):
        if event_type in [
            EventType.TASK_RELEASE,
            EventType.TASK_PLACEMENT,
            EventType.TASK_PREEMPT,
            EventType.TASK_MIGRATION,
            EventType.TASK_FINISHED,
        ]:
            if task is None:
                raise ValueError(f"No task provided with {event_type}")
            if event_type in [EventType.TASK_PLACEMENT, EventType.TASK_MIGRATION]:
                if placement is None:
                    raise ValueError(f"No placement provided with {event_type}")
        if type(time) != EventTime:
            raise ValueError(f"Invalid type for time: {type(time)}")
        self._event_type = event_type
        self._time = time
        self._task = task
        self._placement = placement

    def __lt__(self, other):
        if self.time == other.time:
            return self.event_type < other.event_type
        return self.time < other.time

    def __str__(self):
        if self.task is None:
            return f"Event(time={self.time}, type={self.event_type})"
        else:
            return f"Event(time={self.time}, type={self.event_type}, task={self.task})"

    def __repr__(self):
        return str(self)

    @property
    def time(self) -> EventTime:
        return self._time

    @property
    def event_type(self) -> EventType:
        return self._event_type

    @property
    def task(self) -> Optional[Task]:
        return self._task

    @property
    def placement(self) -> Optional[str]:
        return self._placement


class EventQueue(object):
    """An `EventQueue` provides an abstraction that is used by the simulator
    to add the events into, and retrieve events from according to their
    release time.
    """

    def __init__(self):
        self._event_queue = []

    def add_event(self, event: Event):
        """Add the given event to the queue.

        Args:
            event (`Event`): The event to be added to the queue.
        """
        heapq.heappush(self._event_queue, event)

    def next(self) -> Event:
        """Retrieve the next event from the queue.

        Returns:
            The next event in the queue ordered according to the release time.
        """
        return heapq.heappop(self._event_queue)

    def peek(self) -> Optional[Event]:
        """Peek at the next event in the queue without popping it.

        Returns:
            The next event in the queue ordered according to the release time.
        """
        if len(self._event_queue) == 0:
            return None
        return self._event_queue[0]

    def reheapify(self):
        """Reheapify the current queue.

        This method should be used if any in-place changes have been made to
        the events already inserted into the queue.
        """
        heapq.heapify(self._event_queue)

    def __len__(self) -> int:
        return len(self._event_queue)


class Simulator(object):
    """A `Simulator` simulates the execution of the different tasks in the
    system.

    It starts with a list of tasks generated by the sources at a fixed
    frequency, schedules them using the given instance of the Scheduler,
    ensures their execution on the given set of WorkerPools.

    Args:
        worker_pools (`Sequence[WorkerPool]`): A list of `WorkerPool`s
            available for the simulator to execute the tasks on.
        scheduler (`Type[BaseScheduler]`): A scheduler that implements the
            `BaseScheduler` interface, and is used by the simulator to schedule
            the set of available tasks at a regular interval.
        loop_timeout (`EventTime`) [default=sys.maxsize]: The simulator time (in us)
            upto which to run the loop. The default runs until we have exhausted all
            the events in the system.
        scheduler_frequency (`EventTime`) [default=-1]: The time (in us) between two
            subsequent scheduler invocations. The default invokes a new run of
            the scheduler just after the previous one has completed.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        worker_pools: Sequence[WorkerPool],
        scheduler: Type[BaseScheduler],
        workload: Workload,
        loop_timeout: EventTime = EventTime(time=sys.maxsize, unit=EventTime.Unit.US),
        scheduler_frequency: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        if not isinstance(scheduler, BaseScheduler):
            raise ValueError("Scheduler must implement the BaseScheduler interface.")

        if type(loop_timeout) != EventTime:
            raise ValueError(f"Unexpected type of loop_timeout: {type(loop_timeout)}")

        if type(scheduler_frequency) != EventTime:
            raise ValueError(f"Unexpected type of loop_timeout: {type(loop_timeout)}")

        # Set up the logger.
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__, log_file=_flags.csv_file_name
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__, log_file=None
            )

        # Simulator variables.
        self._scheduler = scheduler
        self._workload = workload
        self._workload.populate_task_graphs(loop_timeout)
        self._simulator_time = EventTime(time=0, unit=EventTime.Unit.US)
        self._scheduler_frequency = scheduler_frequency
        self._loop_timeout = loop_timeout

        self._worker_pools = {
            worker_pool.id: worker_pool for worker_pool in worker_pools
        }
        self._logger.info("The Worker Pools are: ")
        for worker_pool in worker_pools:
            self._logger.info(f"{worker_pool}")
            resources_str = ",".join(
                [
                    ",".join((resource.name, resource.id, str(quantity)))
                    for resource, quantity in worker_pool.resources.resources
                ]
            )
            self._csv_logger.debug(
                f"0,WORKER_POOL,{worker_pool.name},{worker_pool.id},{resources_str}"
            )
            for worker in worker_pool.workers:
                self._logger.info(f"\t{worker}")
        self.__log_utilization(self._simulator_time)

        # Internal data.
        self._last_scheduler_start_time = self._simulator_time
        self._next_scheduler_event = None
        self._last_task_placement = []
        self._finished_tasks = 0
        self._missed_deadlines = 0
        self._scheduler_delay = EventTime(
            _flags.scheduler_delay if _flags else 1, EventTime.Unit.US
        )
        self._runtime_variance = _flags.runtime_variance if _flags else 0

        # Initialize the event queue, and add a SIMULATOR_START task to
        # signify the beginning of the simulator loop. Also add a
        # SCHEDULER_START event to invoke the scheduling loop.
        self._event_queue = EventQueue()

        sim_start_event = Event(
            event_type=EventType.SIMULATOR_START, time=self._simulator_time
        )
        self._event_queue.add_event(sim_start_event)
        self._logger.info(
            f"[{self._simulator_time}] Added {sim_start_event} to the event queue."
        )

        sched_start_event = Event(
            event_type=EventType.SCHEDULER_START, time=self._simulator_time
        )
        self._event_queue.add_event(sched_start_event)
        self._logger.info(
            f"[{self._simulator_time}] Added {sched_start_event} to the event queue."
        )

    def simulate(self) -> None:
        """Run the simulator loop.

        This loop requires the `Workload` to be populated with the `TaskGraph`s whose
        execution is to be simulated using the Scheduler.
        """
        # Retrieve the set of released tasks from the graph.
        # At the beginning, this should consist of all the sensor tasks
        # that we expect to run during the execution of the workload,
        # along with their expected release times.
        for task in self._workload.release_tasks():
            event = Event(
                event_type=EventType.TASK_RELEASE,
                time=task.release_time,
                task=task,
            )
            self._event_queue.add_event(event)
            self._logger.info(
                f"[{self._simulator_time}] Added {event} for "
                f"{task} from {task.task_graph} to the event queue."
            )

        # Run the simulator loop.
        while True:
            # If there are any running tasks, step through the execution of
            # the Simulator until the closest remaining time.
            running_tasks = []
            for worker_pool in self._worker_pools.values():
                running_tasks.extend(worker_pool.get_placed_tasks())

            if len(running_tasks) > 0:
                # There are running tasks, figure out the minimum remaining
                # time across all the tasks.
                min_task_remaining_time = min(
                    map(attrgetter("remaining_time"), running_tasks)
                )
                time_until_next_event = (
                    self._event_queue.peek().time - self._simulator_time
                )

                # If the minimum remaining time comes before the time until
                # the next event in the queue, step all workers until the
                # completion of that task, otherwise, handle the next event.
                if min_task_remaining_time < time_until_next_event:
                    self.__step(step_size=min_task_remaining_time)
                else:
                    if self.__handle_event(self._event_queue.next()):
                        break
            else:
                if self.__handle_event(self._event_queue.next()):
                    break

    def __handle_scheduler_start(self, event: Event):
        # Log the required CSV information.
        currently_placed_tasks = []
        for worker_pool in self._worker_pools.values():
            currently_placed_tasks.extend(worker_pool.get_placed_tasks())
        schedulable_tasks = self._workload.get_schedulable_tasks(
            event.time, self._scheduler.lookahead, self._scheduler.preemptive
        )
        self._csv_logger.debug(
            f"{event.time.time},SCHEDULER_START,{len(schedulable_tasks)},"
            f"{len(currently_placed_tasks)}"
        )

        # Execute the scheduler, and insert an event notifying the
        # end of the scheduler into the loop.
        self._last_scheduler_start_time = event.time
        self._next_scheduler_event = None
        self._logger.info(
            f"[{event.time}] Running the scheduler with "
            f"{len(schedulable_tasks)} schedulable tasks and "
            f"{len(currently_placed_tasks)} tasks already placed across "
            f"{len(self._worker_pools)} worker pools."
        )
        sched_finished_event = self.__run_scheduler(event)
        self._event_queue.add_event(sched_finished_event)
        self._logger.info(
            f"[{event.time}] Added {sched_finished_event} to the event queue."
        )

    def __handle_scheduler_finish(self, event: Event):
        # Place the tasks on the assigned worker pool, and reset the
        # available events to the tasks that could not be placed.
        self._logger.info(
            f"[{event.time}] Finished executing the scheduler initiated at "
            f"{self._last_scheduler_start_time}. Placing tasks."
        )

        # Log the required CSV information.
        num_placed = len(
            list(filter(lambda p: p[1] is not None, self._last_task_placement))
        )
        num_unplaced = len(self._last_task_placement) - num_placed
        self._csv_logger.debug(
            f"{event.time.time},SCHEDULER_FINISHED,"
            f"{event.time - self._last_scheduler_start_time},"
            f"{num_placed},{num_unplaced}"
        )

        placement_events = []
        for task, placement, start_time in self._last_task_placement:
            if task.is_complete():
                # Task completd before the scheduler finished.
                continue
            if placement is None:
                if task.worker_pool_id:
                    self._logger.info(f"[{event.time}] Task {task} was preempted")
                    placement_events.append(
                        Event(
                            event_type=EventType.TASK_PREEMPT,
                            time=event.time,
                            task=task,
                        )
                    )
                else:
                    # Task was not placed.
                    self._csv_logger.debug(
                        f"{event.time},TASK_SKIP,{task.name},{task.timestamp},{task.id}"
                    )
                    self._logger.warning(f"[{event.time}] Failed to place {task}")
            elif task.worker_pool_id == placement:
                # Task remained on the same worker pool.
                pass
            elif task.worker_pool_id is None:
                placement_events.append(
                    Event(
                        event_type=EventType.TASK_PLACEMENT,
                        time=max(start_time, event.time),
                        task=task,
                        placement=placement,
                    )
                )
            else:
                # Preempt all the tasks first so the resources are cleared.
                placement_events.append(
                    Event(
                        event_type=EventType.TASK_PREEMPT,
                        time=event.time,
                        task=task,
                    )
                )
                placement_events.append(
                    Event(
                        event_type=EventType.TASK_MIGRATION,
                        time=max(start_time, event.time),
                        task=task,
                        placement=placement,
                    )
                )

        # Sort the events so that preemptions and migrations happen first.
        for placement_event in sorted(placement_events):
            self._event_queue.add_event(placement_event)

        # Reset the available tasks and the last task placement.
        self._last_task_placement = []

        # The scheduler has finished its execution, insert an event for the next
        # invocation of the scheduler.
        next_sched_event = self.__get_next_scheduler_event(
            event,
            self._scheduler_frequency,
            self._last_scheduler_start_time,
            self._loop_timeout,
        )
        self._event_queue.add_event(next_sched_event)
        self._logger.info(
            f"[{event.time}] Added {next_sched_event} to the event queue."
        )

        # Now that all the tasks are placed, ask the simulator to log the resource
        # utilization and quit later, if requested.
        self._event_queue.add_event(
            Event(event_type=EventType.LOG_UTILIZATION, time=event.time)
        )

    def __handle_task_release(self, event: Event):
        # Release a task for the scheduler.
        self._logger.info(
            f"[{event.time}] Added the task from {event} to the released tasks."
        )
        self._csv_logger.debug(
            f"{event.time.time},TASK_RELEASE,{event.task.name},"
            f"{event.task.timestamp},{event.task.intended_release_time},"
            f"{event.task.release_time},{event.task.runtime},"
            f"{event.task.deadline},{event.task.id}"
        )
        # If we are not in the midst of a scheduler invocation and next
        # scheduled invocation is too late, then bring the invocation sooner
        # (event time + scheduler_delay), and re-heapify the event queue.
        if self._next_scheduler_event:
            new_scheduler_event_time = min(
                self._next_scheduler_event.time, event.time + self._scheduler_delay
            )
            if self._next_scheduler_event._time != new_scheduler_event_time:
                self._logger.info(
                    f"[{event.time}] While adding the task from{event}, the scheduler "
                    f"event from {self._next_scheduler_event._time} was pulled back "
                    f"to {new_scheduler_event_time}."
                )
                self._next_scheduler_event._time = new_scheduler_event_time
                self._event_queue.reheapify()

    def __handle_task_finished(self, event: Event):
        self._finished_tasks += 1
        self._csv_logger.debug(
            f"{event.time.time},TASK_FINISHED,{event.task.name},{event.task.timestamp},"
            f"{event.task.completion_time},{event.task.deadline},{event.task.id}"
        )

        # Log if the task missed its deadline or not.
        if event.time > event.task.deadline:
            self._missed_deadlines += 1
            self._csv_logger.debug(
                f"{event.time.time},MISSED_DEADLINE,{event.task.name},"
                f"{event.task.timestamp},{event.task.deadline},{event.task.id}"
            )

        # The given task has finished execution, unlock dependencies from the `Workload`
        new_tasks = self._workload.notify_task_completion(event.task, event.time)
        self._logger.info(
            f"[{event.time}] Notified the Workload of the completion of {event.task} "
            f"from {event.task.task_graph}, and received {len(new_tasks)} new tasks."
        )

        # Add events corresponding to the dependencies.
        for index, task in enumerate(new_tasks, start=1):
            event = Event(
                event_type=EventType.TASK_RELEASE,
                time=task.release_time,
                task=task,
            )
            self._event_queue.add_event(event)
            self._logger.info(
                f"[{event.time}] ({index}/{len(new_tasks)}) "
                f"Added {event} for {task} to the event queue."
            )

    def __handle_task_preempt(self, event: Event):
        task = event.task
        self._csv_logger.debug(
            f"{event.time.time},TASK_PREEMPT,{task.name},{task.timestamp},{task.id}"
        )
        worker_pool = self._worker_pools[task.worker_pool_id]
        worker_pool.remove_task(task)
        task.preempt(event.time)

    def __handle_task_placement(self, event: Event):
        task = event.task
        if not task.is_ready_to_run():
            # TODO: We might want to cache this placement and apply it when the
            # task is released.
            self._csv_logger.debug(
                f"{event.time.time},TASK_NOT_READY,{task.name},{task.timestamp},"
                f"{task.id},{event.placement}"
            )
            return
        # Initialize the task at the given placement time, and place it on
        # the WorkerPool.
        worker_pool = self._worker_pools[event.placement]
        success = worker_pool.place_task(task)
        if success:
            task.start(
                event.time,
                worker_pool_id=event.placement,
                variance=self._runtime_variance,
            )
            resource_allocation_str = ",".join(
                [
                    ",".join((resource.name, resource.id, str(quantity)))
                    for resource, quantity in worker_pool.get_allocated_resources(task)
                ]
            )

            self._csv_logger.debug(
                f"{event.time.time},TASK_PLACEMENT,{task.name},{task.timestamp},"
                f"{task.id},{event.placement},{resource_allocation_str}"
            )
            self._logger.info(f"[{event.time}] Placed {task} on {worker_pool}")
        else:
            self._logger.warning(
                f"[{event.time}] Task {task} cannot be placed on worker {worker_pool}"
            )

    def __handle_task_migration(self, event: Event):
        task = event.task
        assert (
            task.state == TaskState.PREEMPTED
        ), f"The {task} was not PREEMPTED before being MIGRATED."

        last_preemption = task.last_preemption
        assert last_preemption is not None, f"The {task} did not have a preemption."
        self._logger.info(
            f"[{event.time}] Migrating {task} from {last_preemption.old_worker_pool} "
            f"to {event.placement}."
        )
        self._logger.debug(
            f"[event.time] The resource requirements of {task} were "
            f"{task.resource_requirements}."
        )

        worker_pool = self._worker_pools[event.placement]
        success = worker_pool.place_task(task)
        if success:
            task.resume(event.time, worker_pool_id=event.placement)
            self._logger.debug(
                f"[{event.time}] The state of the "
                f"WorkerPool({event.placement}) is {worker_pool.resources}."
            )
            resource_allocation_str = ",".join(
                [
                    ",".join((resource.name, resource.id, str(quantity)))
                    for resource, quantity in worker_pool.get_allocated_resources(task)
                ]
            )
            self._csv_logger.debug(
                f"{event.time.time},TASK_MIGRATED,{task.name},{task.timestamp},"
                f"{task.id},{last_preemption.old_worker_pool},{event.placement},"
                f"{resource_allocation_str}"
            )
        else:
            self._logger.warning(
                f"[{event.time}] Task {task} cannot be migrated to worker {worker_pool}"
            )

    def __handle_event(self, event: Event) -> bool:
        """Handles the next event from the EventQueue.

        Invoked by the simulator loop, and tested using unit tests.

        Args:
            event (`Event`): The event to handle.

        Returns:
            `True` if the event is a SIMULATOR_END and the simulator loop
            should be stopped, `False` otherwise.
        """
        self._logger.info(f"[{event.time}] Received {event} from the event queue.")
        assert (
            event.time >= self._simulator_time
        ), f"Simulator cannot move time from {self._simulator_time} to {event.time}"
        # Advance the clock until the occurrence of this event.
        self.__step(step_size=event.time - self._simulator_time)

        if event.event_type == EventType.SIMULATOR_START:
            # Start of the simulator loop.
            self._csv_logger.debug(
                f"{event.time.time},SIMULATOR_START,{len(self._workload)}"
            )
            self._logger.info(f"Starting the simulator loop at time {event.time}")
        elif event.event_type == EventType.SIMULATOR_END:
            # End of the simulator loop.
            self._csv_logger.debug(
                f"{event.time.time},SIMULATOR_END,{self._finished_tasks},"
                f"{self._missed_deadlines}"
            )
            self._logger.info(
                f"[{event.time}] Ending the simulator loop at time {event.time}"
            )
            return True
        elif event.event_type == EventType.TASK_RELEASE:
            self.__handle_task_release(event)
        elif event.event_type == EventType.TASK_FINISHED:
            self.__handle_task_finished(event)
        elif event.event_type == EventType.TASK_PREEMPT:
            self.__handle_task_preempt(event)
        elif event.event_type == EventType.TASK_PLACEMENT:
            self.__handle_task_placement(event)
        elif event.event_type == EventType.TASK_MIGRATION:
            self.__handle_task_migration(event)
        elif event.event_type == EventType.SCHEDULER_START:
            self.__handle_scheduler_start(event)
        elif event.event_type == EventType.SCHEDULER_FINISHED:
            self.__handle_scheduler_finish(event)
        elif event.event_type == EventType.LOG_UTILIZATION:
            self.__log_utilization(event.time)
        else:
            raise ValueError(f"[{event.time}] Retrieved event of unknown type: {event}")
        return False

    def __step(self, step_size: EventTime = EventTime(1, EventTime.Unit.US)):
        """Advances the clock by the given `step_size`.

        Args:
            step_size (`EventTime`) [default=1]: The amount by which to advance
                the clock (in us).
        """
        self._logger.info(
            f"[{self._simulator_time}] Stepping for {step_size} timesteps."
        )
        if step_size < EventTime(0, EventTime.Unit.US):
            raise ValueError(f"Simulator cannot step backwards {step_size}")
        completed_tasks = []
        for worker_pool in self._worker_pools.values():
            completed_tasks.extend(worker_pool.step(self._simulator_time, step_size))

        # Add TASK_FINISHED events for all the completed tasks.
        self._simulator_time += step_size
        for task in completed_tasks:
            task_finished_event = Event(
                event_type=EventType.TASK_FINISHED, time=self._simulator_time, task=task
            )
            self._event_queue.add_event(task_finished_event)
            self._logger.info(
                f"[{self._simulator_time}] Added {task_finished_event} to the "
                f"event queue."
            )

    def __get_next_scheduler_event(
        self,
        event: Event,
        scheduler_frequency: EventTime,
        last_scheduler_start_time: EventTime,
        loop_timeout: EventTime = EventTime(sys.maxsize, EventTime.Unit.US),
    ) -> Event:
        """Computes the next event when the scheduler should run.

        This method returns a SIMULATOR_END event if either the loop timeout
        is reached, or there are no future task releases in the event queue
        or released tasks.

        Args:
            event (`Event`): The event at which the last scheduler invocation
                finished.
            scheduler_frequency (`EventTime`): The frequency at which the simulator
                needs to be invoked (in us).
            last_scheduler_start_time (`EventTime`): The time at which the last
                invocation of scheduler occurred.
            loop_timeout (`EventTime`): The time at which the simulator loop is
                required to end.

        Returns:
            An event signifying when the next scheduler invocation should be.
            May be of type SCHEDULER_START or SIMULATOR_END

        Raises:
            `ValueError` if an event type != SCHEDULER_FINISHED is passed.
        """
        if not (event.event_type == EventType.SCHEDULER_FINISHED):
            raise ValueError(f"Incorrect event type {event.event_type} passed.")

        scheduler_start_time = None
        if scheduler_frequency < EventTime(0, EventTime.Unit.US):
            # Insert a new scheduler event for the next step.
            scheduler_start_time = event.time + EventTime(1, EventTime.Unit.US)
        else:
            # Calculate when the next scheduler event was going to
            # occur according to its periodicity.
            next_scheduler_time = last_scheduler_start_time + scheduler_frequency

            # If that time has already occurred, invoke a scheduler
            # in the next time step, otherwise wait until that time.
            if next_scheduler_time < event.time:
                self._logger.warning(
                    f"[{event.time}] The scheduler invocations are late. "
                    f"Supposed to start at {next_scheduler_time}, "
                    f"currently {event.time}"
                )
                scheduler_start_time = event.time + EventTime(1, EventTime.Unit.US)
            else:
                scheduler_start_time = next_scheduler_time
        self._logger.debug(
            f"[{event.time}] Executing the next scheduler at {scheduler_start_time} "
            f"because the frequency was {scheduler_frequency}."
        )

        # End the loop according to the timeout, if reached.
        if scheduler_start_time >= loop_timeout:
            self._logger.info(
                f"[{event.time}] The next scheduler start was scheduled "
                f"at {scheduler_start_time}, but the loop timeout is "
                f"{loop_timeout}. Ending the loop."
            )
            return Event(event_type=EventType.SIMULATOR_END, time=loop_timeout)

        # Find sources of existing or ongoing work in the Simulator.
        running_tasks = []
        for worker_pool in self._worker_pools.values():
            running_tasks.extend(worker_pool.get_placed_tasks())
        schedulable_tasks = self._workload.get_schedulable_tasks(
            event.time, self._scheduler.lookahead, self._scheduler.preemptive
        )
        next_event = self._event_queue.peek()

        # If there is either existing work in the form of events in the queue or tasks
        # waiting to be scheduled, or currently running tasks that can lead to more
        # work, adjust the scheduler invocation time accordingly, or end the loop.
        if (
            next_event is None
            and len(schedulable_tasks) == 0
            and len(running_tasks) == 0
        ):
            self._logger.info(
                f"[{event.time}] There are no currently schedulable tasks, "
                f"no running tasks, and no events available in "
                f"the event queue. Ending the loop."
            )
            return Event(
                event_type=EventType.SIMULATOR_END,
                time=event.time + EventTime(1, EventTime.Unit.US),
            )
        elif (
            len(schedulable_tasks) == 0
            or all(task.state == TaskState.RUNNING for task in schedulable_tasks)
            or all(worker_pool.is_full() for worker_pool in self._worker_pools.values())
        ):
            # If there are no schedulable tasks currently, or all schedulable tasks are
            # already running (in a preemptive scheduling scenario), or the WorkerPool
            # is full, adjust the # scheduler invocation time according to either the
            # time of invocation of # the next event, or the minimum completion time
            # of a running task.
            minimum_running_task_completion_time = (
                self._simulator_time
                + min(
                    map(attrgetter("remaining_time"), running_tasks),
                    default=EventTime(0, EventTime.Unit.US),
                )
                + self._scheduler_delay
            )

            next_event_invocation_time = (
                next_event.time + self._scheduler_delay
                if next_event is not None
                else EventTime(0, EventTime.Unit.US)
            )

            adjusted_scheduler_start_time = max(
                scheduler_start_time,
                minimum_running_task_completion_time,
                next_event_invocation_time,
            )

            if scheduler_start_time != adjusted_scheduler_start_time:
                self._logger.warn(
                    f"[{event.time}] The scheduler start time was pushed from "
                    f"{scheduler_start_time} to {adjusted_scheduler_start_time} since "
                    f"either the next running task finishes at "
                    f"{minimum_running_task_completion_time} or the next event is "
                    f"being invoked at {next_event_invocation_time}."
                )
                scheduler_start_time = adjusted_scheduler_start_time

        # Save the scheduler event in case its start time needs to be pulled
        # back by the arrival of a task.
        self._next_scheduler_event = Event(
            event_type=EventType.SCHEDULER_START, time=scheduler_start_time
        )
        return self._next_scheduler_event

    def __run_scheduler(self, event: Event) -> Event:
        """Run the scheduler.

        Args:
            event (`Event`): The event at which the scheduler was invoked.

        Returns:
            An `Event` signifying the end of the scheduler invocation.

        Raises:
            `ValueError` if an event type != SCHEDULER_START is passed.
        """
        if not (event.event_type == EventType.SCHEDULER_START):
            raise ValueError("Incorrect event type passed.")
        scheduler_runtime, task_placement = self._scheduler.schedule(
            event.time, self._workload, WorkerPools(self._worker_pools.values())
        )
        placement_time = event.time + scheduler_runtime
        self._last_task_placement = task_placement

        return Event(event_type=EventType.SCHEDULER_FINISHED, time=placement_time)

    def __log_utilization(self, sim_time: EventTime):
        """Logs the utilization of the resources of a particular WorkerPool.

        Args:
            sim_time (`EventTime`): The simulation time at which the utilization
                is logged (in us).
        """
        assert (
            sim_time.unit == EventTime.Unit.US
        ), "The simulator time was not in microseconds."

        # Cumulate the resources from all the WorkerPools
        for worker_pool in self._worker_pools.values():
            worker_pool_resources = worker_pool.resources
            for resource_name in set(
                map(lambda value: value[0].name, worker_pool_resources.resources)
            ):
                resource = Resource(name=resource_name, _id="any")
                self._csv_logger.debug(
                    f"{sim_time.time},WORKER_POOL_UTILIZATION,{worker_pool.id},"
                    f"{resource_name},"
                    f"{worker_pool_resources.get_allocated_quantity(resource)},"
                    f"{worker_pool_resources.get_available_quantity(resource)}"
                )
