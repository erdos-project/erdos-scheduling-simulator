import heapq
import logging
import os
import sys
from enum import Enum
from functools import total_ordering
from operator import attrgetter, itemgetter
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401

from data import BaseWorkloadLoader
from schedulers import BaseScheduler
from utils import EventTime, setup_csv_logging, setup_logging
from workers import WorkerPools
from workload import (
    Placement,
    Placements,
    Resource,
    Task,
    TaskGraph,
    TaskState,
    Workload,
)


@total_ordering
class EventType(Enum):
    """Represents the different Events that a simulator has to simulate."""

    SIMULATOR_START = 0  # Signify the start of the simulator loop.
    TASK_CANCEL = 1  # Ask the simulator to cancel the task.
    EVICT_PROFILE = 2  # Ask the simulator to evict the profile from the WorkerPool.
    TASK_FINISHED = 3  # Notify the simulator of the end of a task.
    TASK_GRAPH_RELEASE = 4  # Notify the simulator of the release of a task graph.
    TASK_RELEASE = 5  # Ask the simulator to release the task.
    UPDATE_WORKLOAD = 6  # Ask the simulator to update the workload.
    TASK_PREEMPT = 7  # Ask the simulator to preempt a task.
    TASK_MIGRATION = 8  # Ask the simulator to migrate a task.
    LOAD_PROFILE = 9  # Ask the simulator to load a profile into the WorkerPool.
    TASK_PLACEMENT = 10  # Ask the simulator to place a task.
    SCHEDULER_START = 11  # Requires the simulator to invoke the scheduler.
    SCHEDULER_FINISHED = 12  # Signifies the end of the scheduler loop.
    SIMULATOR_END = 13  # Signify the end of the simulator loop.
    LOG_UTILIZATION = 14  # Ask the simulator to log worker pool utilization.

    def __lt__(self, other) -> bool:
        # This method is used to order events in the event queue. We prioritize
        # events that first free resources.
        return self.value < other.value

    def __eq__(self, other) -> bool:
        return self.value == other.value


class Event(object):
    """Represents an `Event` that is managed by the `EventQueue` and informs
    what action the simulator needs to do at the given simulator time.

    Args:
        event_type (`EventType`): The type of the event.
        time (`EventTime`): The simulator time at which the event occurred.
        task (`Optional[Task]`): The task associated with this event if it is
            of type `TASK_RELEASE` or `TASK_FINISHED`.
        task_graph (`Optional[str]`): The name of the TaskGraph associated with
            this event.

    Raises:
        `ValueError` if the event is of type `TASK_RELEASE` or `TASK_FINISHED`
        and no associated task is provided, or if the time is not of type `EventTime`.
    """

    def __init__(
        self,
        event_type: EventType,
        time: EventTime,
        task: Optional[Task] = None,
        task_graph: Optional[str] = None,
        placement: Optional[Placement] = None,
    ):
        if event_type in [
            EventType.TASK_CANCEL,
            EventType.TASK_RELEASE,
            EventType.TASK_PLACEMENT,
            EventType.TASK_PREEMPT,
            EventType.TASK_MIGRATION,
            EventType.TASK_FINISHED,
        ]:
            if task is None:
                raise ValueError(f"No task provided with {event_type}.")
            if event_type in [EventType.TASK_PLACEMENT, EventType.TASK_MIGRATION]:
                if placement is None:
                    raise ValueError(f"No placement provided with {event_type}.")
                elif type(placement) != Placement:
                    raise ValueError(f"Invalid type for placement: {type(placement)}.")

        if event_type == EventType.TASK_GRAPH_RELEASE and task_graph is None:
            raise ValueError(f"No task graph provided with {event_type}.")

        if type(time) != EventTime:
            raise ValueError(f"Invalid type for time: {type(time)}")
        self._event_type = event_type
        self._time = time
        self._task = task
        self._task_graph = (
            task.task_graph if task is not None and task_graph is None else task_graph
        )
        self._placement = placement

    def __lt__(self, other: "Event") -> bool:
        if self.time == other.time:
            if (
                self.event_type == other.event_type
                and self.task is not None
                and other.task is not None
            ):
                return self.task.unique_name < other.task.unique_name
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
    def task_graph(self) -> Optional[str]:
        return self._task_graph

    @property
    def placement(self) -> Optional[Placement]:
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

    def remove_event(self, event: Event):
        """Removes the event from the queue.

        Args:
            event (`Event`): The event to be removed from the queue.

        Raises:
            `ValueError` if the event was not found.
        """
        self._event_queue.remove(event)
        self.reheapify()

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

    def get_next_event_of_type(self, event_type: EventType) -> Optional[Event]:
        """Retrieve the next event of the given type from the queue.

        Note that this method iterates over the entire queue, and can be slow.

        Args:
            event_type (`EventType`): The type of the event to retrieve.

        Returns:
            The next event of the given type in the queue ordered according to the
            release time.
        """
        filtered_values = list(
            filter(lambda event: event.event_type == event_type, self._event_queue)
        )
        return min(filtered_values) if len(filtered_values) > 0 else None

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
        worker_pools (`WorkerPools`): A `WorkerPools` instance describing
            the worker pools available for the simulator to execute the tasks on.
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
        worker_pools: WorkerPools,
        scheduler: BaseScheduler,
        workload_loader: BaseWorkloadLoader,
        loop_timeout: EventTime = EventTime(time=sys.maxsize, unit=EventTime.Unit.US),
        scheduler_frequency: EventTime = EventTime(time=-1, unit=EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        if not isinstance(scheduler, BaseScheduler):
            raise ValueError("Scheduler must implement the BaseScheduler interface.")

        if not isinstance(workload_loader, BaseWorkloadLoader):
            raise ValueError(
                "WorkloadLoader must implement the BaseWorkloadLoader interface."
            )

        if type(loop_timeout) != EventTime:
            raise ValueError(f"Unexpected type of loop_timeout: {type(loop_timeout)}")

        if type(scheduler_frequency) != EventTime:
            raise ValueError(
                f"Unexpected type of scheduler_frequency: {type(scheduler_frequency)}"
            )

        # Set up the logger.
        # Amended from https://tinyurl.com/da5jfm58
        def event_representation_filter(record):
            if record.args:
                args = []
                for arg in record.args:
                    if isinstance(arg, Event):
                        arg = repr(arg)
                    else:
                        arg = str(arg)
                    args.append(arg)
                record.args = tuple(args)
            return True

        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.csv_file_name,
            )
            self._log_dir = _flags.log_dir
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__, log_file=None
            )
            self._log_dir = os.getcwd()
        if not self._logger.isEnabledFor(logging.DEBUG):
            self._logger.addFilter(event_representation_filter)

        # Simulator variables.
        self._scheduler = scheduler
        self._simulator_time = EventTime.zero()
        self._scheduler_frequency = scheduler_frequency
        self._loop_timeout = loop_timeout
        self._task_id_added_to_event_queue = set()

        # Workload variables.
        self._workload_loader = workload_loader
        self._workload: Workload = Workload.empty(_flags)

        self._worker_pools = worker_pools
        self._logger.info("The Worker Pools are: ")
        for worker_pool in self._worker_pools.worker_pools:
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
        self._last_scheduler_placements: Optional[Placements] = None

        # A Cache from the TaskID to a future Placement event in the EventQueue.
        # The Simulator uses this bookkeeping to revoke / invalidate decisions made
        # by the past scheduler invocations.
        self._future_placement_events: Mapping[str, Placement] = {}

        # Flag values.
        self._scheduler_delay = EventTime(
            _flags.scheduler_delay if _flags else 0, EventTime.Unit.US
        )
        self._runtime_variance = _flags.runtime_variance if _flags else 0
        self._drop_skipped_tasks = _flags.drop_skipped_tasks if _flags else False
        self._verify_schedule = _flags.verify_schedule if _flags else False
        self._run_scheduler_at_worker_free = (
            _flags.scheduler_run_at_worker_free if _flags else False
        )
        self._workload_update_interval = (
            EventTime(_flags.workload_update_interval, EventTime.Unit.US)
            if _flags
            else EventTime.invalid()
        )
        self._log_task_graphs = _flags.log_graphs if _flags else False

        # Statistics about the Task.
        self._finished_tasks = 0
        self._cancelled_tasks = 0
        self._missed_task_deadlines = 0

        # Statistics about the TaskGraph.
        self._finished_task_graphs = 0
        self._missed_task_graph_deadlines = 0

        # Initialize the event queue.
        # To make the system continue working the loop, we add three events:
        # - SIMULATOR_START: A notional event start the simulator and log into the CSV.
        # - UPDATE_WORKLOAD: An event to reach out to the WorkloadLoader and get the
        #   next batch of TaskGraphs.
        # - SCHEDULER_START: An event to invoke the scheduler.
        self._event_queue = EventQueue()

        # First, create the SIMULATOR_START event to signify the start of the Simulator.
        sim_start_event = Event(
            event_type=EventType.SIMULATOR_START, time=self._simulator_time
        )
        self._event_queue.add_event(sim_start_event)
        self._logger.info(
            "[%s] Added %s to the event queue.",
            self._simulator_time.time,
            sim_start_event,
        )

        # Second, create the UPDATE_WORKLOAD event to retrieve the latest Workload.
        upate_workload_event = Event(
            event_type=EventType.UPDATE_WORKLOAD, time=self._simulator_time
        )
        self._event_queue.add_event(upate_workload_event)
        self._logger.info(
            "[%s] Added %s to the event queue.",
            self._simulator_time.time,
            upate_workload_event,
        )

        # Third, create the SCHEDULER_START event to invoke the scheduler.
        sched_start_event = Event(
            event_type=EventType.SCHEDULER_START, time=self._simulator_time
        )
        self._event_queue.add_event(sched_start_event)
        self._logger.info(
            "[%s] Added %s to the event queue.",
            self._simulator_time.time,
            sched_start_event,
        )

    def dry_run(self) -> None:
        """Displays the order in which the TaskGraphs will be released."""
        start_time = EventTime.zero()
        while True:
            # Get the next Workload from the WorkloadLoader.
            next_workload = self._workload_loader.get_next_workload(start_time)
            if next_workload is None:
                self._logger.info(
                    f"The WorkloadLoader '{type(self._workload_loader).__name__}' "
                    f"released no more Workloads."
                )
                break

            # A new Workload has been released, we log the release times of the
            # TaskGraphs from this instance of the Workload.
            self._workload = next_workload
            task_graphs = list(
                sorted(
                    self._workload.task_graphs.values(),
                    key=lambda task_graph: task_graph.release_time,
                )
            )
            self._logger.info(
                f"The WorkloadLoader '{type(self._workload_loader).__name__}' released "
                f"a Workload with {len(task_graphs)} TaskGraphs."
            )
            start_time = task_graphs[-1].release_time

            for task_graph in task_graphs:
                self._logger.info(
                    "[%s] The TaskGraph %s will be released with deadline "
                    "%s and critical path runtime %s.",
                    task_graph.release_time.to(EventTime.Unit.US).time,
                    task_graph.name,
                    task_graph.deadline,
                    task_graph.critical_path_runtime,
                )
                self._csv_logger.info(
                    "%s,TASK_GRAPH_RELEASE,%s,%s,%s,%s",
                    task_graph.release_time.to(EventTime.Unit.US).time,
                    task_graph.name,
                    task_graph.deadline.to(EventTime.Unit.US).time,
                    len(task_graph.get_nodes()),
                    task_graph.critical_path_runtime.to(EventTime.Unit.US).time,
                )
                if self._log_task_graphs:
                    # Log a DOT representation of the TaskGraph, if requested.
                    task_graph.to_dot(
                        os.path.join(self._log_dir, f"{task_graph.name}.dot")
                    )

    def simulate(self) -> None:
        """Run the simulator loop.

        This loop requires the `Workload` to be populated with the `TaskGraph`s whose
        execution is to be simulated using the Scheduler.
        """
        # Run the simulator loop.
        while True:
            time_until_next_event = self._event_queue.peek().time - self._simulator_time

            # If there are any running tasks, step through the execution of the
            # Simulator until the closest remaining time.
            running_tasks = self._worker_pools.get_placed_tasks()

            if len(running_tasks) > 0:
                # There are running tasks, figure out the minimum remaining
                # time across all the tasks.
                min_task_remaining_time = min(
                    map(attrgetter("remaining_time"), running_tasks)
                )
                self._logger.debug(
                    "[%s] The minimum task remaining time was %s, "
                    "and the time until next event was %s.",
                    self._simulator_time.to(EventTime.Unit.US).time,
                    min_task_remaining_time,
                    time_until_next_event,
                )

                # If the minimum remaining time comes before the time until
                # the next event in the queue, step all workers until the
                # completion of that task, otherwise, handle the next event.
                if min_task_remaining_time < time_until_next_event:
                    self.__step(step_size=min_task_remaining_time)
                else:
                    # NOTE: We step here so that all the Tasks that are going
                    # to finish as a result of this step have their TASK_FINISHED
                    # events processed first before any future placement occurs
                    # that is decided prior.
                    self.__step(step_size=time_until_next_event)
                    if self.__handle_event(self._event_queue.next()):
                        break
            else:
                # Step until the next event is supposed to be executed.
                self.__step(step_size=time_until_next_event)
                if self.__handle_event(self._event_queue.next()):
                    break

    def __handle_scheduler_start(self, event: Event) -> None:
        """Handle the SCHEDULER_START event. The method invokes the scheduler, and adds
        a SCHEDULER_FINISHED event to the event queue.

        Args:
            event (`Event`): The event to handle.
        """
        # Log the required CSV information.
        currently_placed_tasks = self._worker_pools.get_placed_tasks()
        schedulable_tasks = self._workload.get_schedulable_tasks(
            event.time,
            self._scheduler.lookahead,
            self._scheduler.preemptive,
            self._scheduler.retract_schedules,
            self._worker_pools,
            self._scheduler.policy,
            self._scheduler.branch_prediction_accuracy,
            self._scheduler.release_taskgraphs,
        )
        self._csv_logger.debug(
            f"{event.time.time},SCHEDULER_START,{len(schedulable_tasks)},"
            f"{len(currently_placed_tasks)}"
        )
        self.__log_utilization(event.time)

        # Execute the scheduler, and insert an event notifying the
        # end of the scheduler into the loop.
        self._last_scheduler_start_time = event.time
        self._next_scheduler_event = None
        self._logger.info(
            "[%s] Running the scheduler with %s schedulable tasks and "
            "%s tasks already placed across %s worker pools.",
            event.time.time,
            len(schedulable_tasks),
            len(currently_placed_tasks),
            len(self._worker_pools),
        )
        sched_finished_event = self.__run_scheduler(event)
        self._event_queue.add_event(sched_finished_event)
        self._logger.info(
            "[%s] Added %s to the event queue.",
            event.time.time,
            sched_finished_event,
        )

    def __handle_profile_eviction(self, event: Event) -> None:
        """Handles the eviction of `WorkProfile`s from the given `WorkerPool`.

        Args:
            event (`Event`): The event that contains the `placement` information with
                the type `Placement.PlacementType.EVICT_WORK_PROFILE`.
        """
        if event.placement.placement_type != Placement.PlacementType.EVICT_WORK_PROFILE:
            raise ValueError(
                f"The placement type of the event {event.placement.placement_type} is "
                f"not {Placement.PlacementType.EVICT_WORK_PROFILE}."
            )
        self._logger.info(
            "[%s] The Simulator is requesting the eviction of %s "
            "from the WorkerPool %s and the Worker %s.",
            event.time.to(EventTime.Unit.US).time,
            event.placement.work_profile,
            event.placement.worker_pool_id,
            event.placement.worker_id,
        )
        worker_pool = self._worker_pools.get_worker_pool(event.placement.worker_pool_id)
        if worker_pool is None:
            raise ValueError(
                f"The WorkerPool {event.placement.worker_pool_id} does not exist."
            )
        else:
            worker_pool.evict_profile(
                event.placement.work_profile,
                event.placement.worker_id,
            )

    def __handle_profile_loading(self, event: Event) -> None:
        """Handles the loading of a `WorkProfile` into the given `WorkerPool`.

        Args:
            event (`Event`): The event that contains the `placement` information with
                the type `Placement.PlacementType.LOAD_WORK_PROFILE`.
        """
        if event.placement.placement_type != Placement.PlacementType.LOAD_WORK_PROFILE:
            raise ValueError(
                f"The placement type of the event {event.placement.placement_type} is "
                f"not {Placement.PlacementType.LOAD_WORK_PROFILE}."
            )
        self._logger.info(
            "[%s] The Simulator is requesting the loading of %s "
            "into the WorkerPool %s and the Worker %s.",
            event.time.to(EventTime.Unit.US).time,
            event.placement.work_profile,
            event.placement.worker_pool_id,
            event.placement.worker_id,
        )
        worker_pool = self._worker_pools.get_worker_pool(event.placement.worker_pool_id)
        if worker_pool is None:
            raise ValueError(
                f"The WorkerPool {event.placement.worker_pool_id} does not exist."
            )
        else:
            worker_pool.load_profile(
                event.placement.work_profile,
                event.placement.loading_strategy,
                event.placement.worker_id,
            )

    def __create_events_from_task_placement_skip(
        self,
        time: EventTime,
        placement: Placement,
        drop_skipped_tasks: bool = False,
    ) -> Sequence[Event]:
        """Handles the non-placement of Tasks by the Scheduler.

        If the simulator was required to drop tasks that were not placed, it returns
        a sequence of `TASK_CANCEL` Events for`Task`s that were cancelled as a result
        of the non-placement of this task, and delays the execution of the Task to the
        next available timestamp otherwise.

        Args:
            time (`EventTime`): The time at which the skipping is being requested.
            placement (`Placement`): The Placement object returned by a Scheduler.
            drop_skipped_tasks (`bool`): If `True`, the cancellation of the `Task` and
                its dependents will be requested.

        Returns:
            A (possibly empty) sequence of cancellation events for `Task`s.
        """
        assert (
            not placement.is_placed()
        ), f"Skipping requested for a placed Task {placement.task.unique_name}."
        self._logger.debug(
            "[%s] Creating events from the skipping of %s, "
            "with drop_skipped_tasks set to %s.",
            time.to(EventTime.Unit.US).time,
            placement.task.unique_name,
            drop_skipped_tasks,
        )

        task_events = []
        if drop_skipped_tasks:
            # The configuration requires us to drop tasks that were skipped, we request
            # the Simulator to cancel this Task along with all of its dependents that
            # cannot be executed as a result.
            task_graph = self._workload.get_task_graph(placement.task.task_graph)
            if task_graph is None:
                raise ValueError(f"No TaskGraph found for {placement.task.task_graph}")

            for cancelled_task in task_graph.cancel(placement.task, time):
                task_events.append(
                    Event(
                        event_type=EventType.TASK_CANCEL,
                        time=cancelled_task.cancellation_time,
                        task=cancelled_task,
                    )
                )

            if task_graph.is_cancelled():
                released_tasks_from_new_task_graph = (
                    self._workload.notify_task_graph_completion(
                        task_graph,
                        time,
                    )
                )
                self._logger.info(
                    "[%s] Notified the Workload of the cancellation of %s, "
                    "and received %s new Tasks from new TaskGraphs.",
                    time.to(EventTime.Unit.US).time,
                    task_graph.name,
                    len(released_tasks_from_new_task_graph),
                )
                for released_task in released_tasks_from_new_task_graph:
                    task_events.append(
                        Event(
                            event_type=EventType.TASK_RELEASE,
                            time=released_task.release_time,
                            task=released_task,
                        )
                    )
        else:
            self._csv_logger.debug(
                f"{time.to(EventTime.Unit.US).time},TASK_SKIP,{placement.task.name},"
                f"{placement.task.task_graph},{placement.task.timestamp},"
                f"{placement.task.id}"
            )
            if placement.task.id in self._future_placement_events:
                future_placement_event = self._future_placement_events[
                    placement.task.id
                ]
                self._logger.warning(
                    "[%s] Failed to place %s, skipping it for future reconsideration "
                    "and invalidating the previously scheduled Placement event for %s.",
                    time.to(EventTime.Unit.US).time,
                    placement.task,
                    future_placement_event.time,
                )
                self._event_queue.remove_event(future_placement_event)
                del self._future_placement_events[placement.task.id]

                # Unschedule the Task.
                placement.task.unschedule(time)
            else:
                self._logger.warning(
                    "[%s] Failed to place %s, skipping it for future reconsideration.",
                    time.to(EventTime.Unit.US).time,
                    placement.task,
                )
        return task_events

    def __create_events_from_task_placement(
        self, event_time: EventTime, placement: Placement
    ) -> Sequence[Event]:
        """Handle the placement of a Task by the Scheduler.

        Note that the `Placement` must be of type `Placement.PlacementType.PLACE_TASK`.

        Args:
            event_time (`EventTime`): The time at which the placement was received.
            placement (`Placement`): The Placement object returned by a Scheduler.

        Returns:
            A sequence of events that need to be added to the event queue as a result
            of this Placement.
        """
        if placement.placement_type != Placement.PlacementType.PLACE_TASK:
            raise ValueError(
                f"Invalid placement type {placement.placement_type} for "
                f"{placement.task.unique_name}."
            )

        simulator_events = []
        if placement.task.state < TaskState.SCHEDULED:
            if placement.is_placed():
                # The Task has not been scheduled before, we populate the
                # required fields and cache the placement event in case
                # changes need to be made before the task is actually executed.
                self._logger.debug(
                    "[%s] Scheduling the Task %s for the first time to be "
                    "executed on WorkerPool %s at %s.",
                    event_time.to(EventTime.Unit.US).time,
                    placement.task,
                    placement.worker_pool_id,
                    placement.placement_time,
                )
                placement.task.schedule(event_time, placement)

                simulator_event = Event(
                    event_type=EventType.TASK_PLACEMENT,
                    time=placement.placement_time,
                    task=placement.task,
                    placement=placement,
                )
                self._future_placement_events[placement.task.id] = simulator_event
                simulator_events.append(simulator_event)
            else:
                simulator_events.extend(
                    self.__create_events_from_task_placement_skip(
                        event_time, placement, self._drop_skipped_tasks
                    )
                )
        elif placement.task.state == TaskState.SCHEDULED:
            if placement.is_placed():
                # The Task was SCHEDULED previously, we update any changes to
                # both the Task and any future placement events in the queue.
                if placement.task.id not in self._future_placement_events:
                    self._logger.debug(
                        "[%s] The Task %s in %s state did not have any "
                        "corresponding cached Placement events in the future."
                        "Rescheduling the Task to be executed on the "
                        "WorkerPool %s at %s.",
                        event_time.to(EventTime.Unit.US).time,
                        placement.task,
                        placement.task.state,
                        placement.worker_pool_id,
                        placement.placement_time,
                    )
                    placement.task.schedule(event_time, placement)
                    simulator_event = Event(
                        event_type=EventType.TASK_PLACEMENT,
                        time=placement.placement_time,
                        task=placement.task,
                        placement=placement,
                    )
                    self._future_placement_events[placement.task.id] = simulator_event
                    simulator_events.append(simulator_event)
                else:
                    cached_placement_event = self._future_placement_events[
                        placement.task.id
                    ]
                    self._logger.debug(
                        "[%s] Updating a prior placement of Task %s from "
                        "WorkerPool %s and time %s to WorkerPool %s and time %s.",
                        event_time.to(EventTime.Unit.US).time,
                        placement.task,
                        cached_placement_event.placement.worker_pool_id,
                        cached_placement_event.time,
                        placement.worker_pool_id,
                        placement.placement_time,
                    )
                    placement.task.schedule(event_time, placement)
                    cached_placement_event._time = placement.placement_time
                    cached_placement_event._placement = placement
                    # Reheapify since we changed the time of some already cached events.
                    self._event_queue.reheapify()
            else:
                simulator_events.extend(
                    self.__create_events_from_task_placement_skip(
                        event_time, placement, self._drop_skipped_tasks
                    )
                )
        elif placement.task.state == TaskState.RUNNING:
            if placement.is_placed():
                if (
                    placement.worker_pool_id != placement.task.worker_pool_id
                    or placement.placement_time > event_time
                ):
                    self._logger.debug(
                        "[%s] The Task %s in state %s was migrated from "
                        "WorkerPool %s to WorkerPool %s to be started at %s.",
                        event_time.to(EventTime.Unit.US).time,
                        placement.task,
                        placement.task.state,
                        placement.task.worker_pool_id,
                        placement.worker_pool_id,
                        placement.placement_time,
                    )
                    simulator_events.append(
                        Event(
                            event_type=EventType.TASK_PREEMPT,
                            time=event_time,
                            task=placement.task,
                        )
                    )
                    simulator_events.append(
                        Event(
                            event_type=EventType.TASK_MIGRATION,
                            time=placement.placement_time,
                            task=placement.task,
                            placement=placement,
                        )
                    )
            else:
                self._logger.debug(
                    "[%s] The Task %s in state %s was preempted.",
                    event_time.to(EventTime.Unit.US).time,
                    placement.task,
                    placement.task.state,
                )
                simulator_events.append(
                    Event(
                        event_type=EventType.TASK_PREEMPT,
                        time=event_time,
                        task=placement.task,
                    )
                )
        elif placement.task.state == TaskState.PREEMPTED:
            raise NotImplementedError(
                "Rescheduling of PREEMPTED tasks hasn't been implemented yet."
            )
        else:
            if not placement.is_placed():
                self._logger.debug(
                    "[%s] The Task %s was cancelled by an upstream task, "
                    "skipping its re-cancellation.",
                    event_time.to(EventTime.Unit.US).time,
                    placement.task,
                )
            else:
                # Task was either completed or cancelled before the Scheduler finished,
                # we skip the application of this Placement decision.
                self._logger.warning(
                    "[%s] Skipping the application of Placement of Task %s at "
                    "time %s because the Task was in %s state.",
                    event_time.to(EventTime.Unit.US).time,
                    placement.task,
                    placement.placement_time,
                    placement.task.state,
                )

        return simulator_events

    def __handle_scheduler_finish(self, event: Event) -> None:
        """Handle the SCHEDULER_FINISHED event. The method places the profiles and tasks
        on the appropriate workers, and computes the next scheduler start time.

        Args:
            event (`Event`): The event to handle.
        """
        # Place the tasks on the assigned worker pool, and reset the
        # available events to the tasks that could not be placed.
        self._logger.info(
            "[%s] Finished executing the scheduler initiated at %s. Placing tasks.",
            event.time.time,
            self._last_scheduler_start_time,
        )

        # Log the required CSV information.
        def count_placed_tasks(placements: Placements):
            return len(
                list(
                    filter(
                        lambda p: p.placement_type == Placement.PlacementType.PLACE_TASK
                        and p.is_placed(),
                        placements,
                    )
                )
            )

        num_placed = count_placed_tasks(self._last_scheduler_placements)
        num_unplaced = count_placed_tasks(self._last_scheduler_placements) - num_placed
        scheduler_runtime = event.time - self._last_scheduler_start_time
        self._csv_logger.debug(
            f"{event.time.time},SCHEDULER_FINISHED,"
            f"{scheduler_runtime.to(EventTime.Unit.US).time},"
            f"{num_placed},{num_unplaced},"
            f"{self._last_scheduler_placements.true_runtime.to(EventTime.Unit.US).time}"
        )

        if self._verify_schedule:
            self._scheduler.verify_schedule(
                event.time,
                self._workload,
                self._worker_pools,
                self._last_scheduler_placements,
            )

        simulator_events = []
        for placement in self._last_scheduler_placements:
            if placement.placement_type == Placement.PlacementType.CANCEL_TASK:
                # Request the cancellation of this `Task` and all its dependent `Task`s
                # that cannot complete their execution without this `Task`.
                simulator_events.extend(
                    self.__create_events_from_task_placement_skip(
                        time=event.time, placement=placement, drop_skipped_tasks=True
                    )
                )
            elif placement.placement_type == Placement.PlacementType.PLACE_TASK:
                # Ensure that the Placement always occurs in the present or future.
                if placement.is_placed() and placement.placement_time < event.time:
                    self._logger.error(
                        "[%s] A Placement for the Task %s occurred in the past at %s.",
                        event.time.to(EventTime.Unit.US).time,
                        placement.task,
                        placement.placement_time,
                    )
                    raise ValueError(
                        f"A Placement for the Task {placement.task.unique_name} "
                        f"occurred in the past at {placement.placement_time}."
                    )
                if placement.is_placed():
                    # If there was a Placement, log the Task scheduling event.
                    self._csv_logger.debug(
                        f"{event.time.time},TASK_SCHEDULED,{placement.task.name},"
                        f"{placement.task.task_graph},{placement.task.timestamp},"
                        f"{placement.task.id},{placement.task.deadline.time},"
                        f"{placement.placement_time.time},{placement.worker_pool_id},"
                        f"{placement.execution_strategy.runtime.time}"
                    )
                simulator_events.extend(
                    self.__create_events_from_task_placement(event.time, placement)
                )
            elif placement.placement_type == Placement.PlacementType.EVICT_WORK_PROFILE:
                # Ensure that the Placement always occurs in the present or future.
                if placement.placement_time < event.time:
                    self._logger.error(
                        "[%s] A Placement for the WorkProfile %s occurred in "
                        "the past at %s.",
                        event.time.to(EventTime.Unit.US).time,
                        placement.work_profile,
                        placement.placement_time,
                    )
                    raise ValueError(
                        f"A Placement for the WorkProfile {placement.work_profile} "
                        f"occurred in the past at {placement.placement_time}."
                    )
                eviction_event = Event(
                    event_type=EventType.EVICT_PROFILE,
                    time=placement.placement_time,
                    placement=placement,
                )
                simulator_events.append(eviction_event)
            elif placement.placement_type == Placement.PlacementType.LOAD_WORK_PROFILE:
                # Ensure that the Placement always occurs in the present or future.
                if placement.placement_time < event.time:
                    self._logger.error(
                        "[%s] A Placement for the WorkProfile %s occurred in "
                        "the past at %s.",
                        event.time.to(EventTime.Unit.US).time,
                        placement.work_profile,
                        placement.placement_time,
                    )
                    raise ValueError(
                        f"A Placement for the WorkProfile {placement.work_profile} "
                        f"occurred in the past at {placement.placement_time}."
                    )
                profile_load_event = Event(
                    event_type=EventType.LOAD_PROFILE,
                    time=placement.placement_time,
                    placement=placement,
                )
                simulator_events.append(profile_load_event)

        # Sort the events so that preemptions and migrations happen first.
        for simulator_event in sorted(simulator_events):
            self._logger.info(
                "[%s] Added %s to the event queue.",
                event.time.time,
                simulator_event,
            )
            self._event_queue.add_event(simulator_event)

        # Reset the available tasks and the last task placement.
        self._last_scheduler_placements = None

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
            "[%s] Added %s to the event queue.", event.time.time, next_sched_event
        )

        # Now that all the tasks are placed, ask the simulator to log the resource
        # utilization and quit later, if requested.
        self._event_queue.add_event(
            Event(event_type=EventType.LOG_UTILIZATION, time=event.time)
        )

    def __handle_task_cancellation(self, event: Event) -> None:
        """Handle the cancellation of a task. If the scheduler decided to cancel the
        task or the conditional task was resolved to never be executed, we cancel the
        Task and remove any future placements queued for this Task.

        Args:
            event (`Event`): The event containing the task to be cancelled.
        """
        self._cancelled_tasks += 1
        self._logger.info(
            "[%s] (%s) The Simulator is cancelling the task %s.",
            event.time.to(EventTime.Unit.US).time,
            self._cancelled_tasks,
            event.task,
        )
        self._csv_logger.debug(
            f"{event.time.to(EventTime.Unit.US).time},TASK_CANCEL,{event.task.name},"
            f"{event.task.timestamp},{event.task.id},{event.task.task_graph},"
            f"{event.task.slowest_execution_strategy.runtime.time}"
        )

        # If the task already had a placement, we remove the placement from our queue.
        if event.task.id in self._future_placement_events:
            placement_event = self._future_placement_events[event.task.id]
            self._logger.debug(
                "[%s] The task %s being cancelled has a future placement for %s.",
                event.time.to(EventTime.Unit.US).time,
                event.task,
                placement_event.time,
            )
            self._event_queue.remove_event(placement_event)
            del self._future_placement_events[event.task.id]

    def __handle_task_release(self, event: Event) -> None:
        """Handle the release of a task. If the task is not already scheduled and the
        next scheduler event is too far into the future, we bring the next scheduler
        event closer to ensure minimal queueing delay.

        Args:
            event (`Event`): The event containing the task to be released.
        """
        # Release a task for the scheduler.
        event.task.release(event.time)

        slowest_execution_strategy = (
            event.task.available_execution_strategies.get_slowest_strategy()
        )
        resources_str = ",".join(
            [
                ",".join((resource.name, resource.id, str(quantity)))
                for resource, quantity in slowest_execution_strategy.resources.resources
            ]
        )
        self._csv_logger.debug(
            f"{event.time.time},TASK_RELEASE,{event.task.name},"
            f"{event.task.timestamp},"
            f"{event.task.intended_release_time.to(EventTime.Unit.US).time},"
            f"{event.task.release_time.to(EventTime.Unit.US).time},"
            f"{event.task.deadline.to(EventTime.Unit.US).time},{event.task.id},"
            f"{event.task.task_graph},"
            f"{slowest_execution_strategy.runtime.to(EventTime.Unit.US).time},"
            f"{resources_str}"
        )

        # If we are not in the midst of a scheduler invocation, and the task hasn't
        # already been scheduled and next scheduled invocation is too late, then
        # bring the invocation sooner to (event time + scheduler_delay), and re-heapify
        # the event queue. We don't do this if the user specifically requested to run
        # the scheduler whenever the `Worker` becomes free.
        if (
            event.task.state < TaskState.SCHEDULED
            and self._next_scheduler_event
            and not self._run_scheduler_at_worker_free
        ):
            new_scheduler_event_time = min(
                self._next_scheduler_event.time, event.time + self._scheduler_delay
            )
            if self._next_scheduler_event._time != new_scheduler_event_time:
                self._logger.info(
                    "[%s] While adding the task from %s, the scheduler event from %s "
                    "was pulled back to %s.",
                    event.time.time,
                    event,
                    self._next_scheduler_event._time,
                    new_scheduler_event_time,
                )
                self._next_scheduler_event._time = new_scheduler_event_time
                self._event_queue.reheapify()

    def __handle_task_finished(self, event: Event) -> None:
        """Handle the completion of a task. The Task is first removed from the Worker
        where it was placed, and then the Task is marked as finished. If the finishing
        of this `Task` completes the `TaskGraph`, then the `TaskGraph` is marked as
        complete and the event is logged. If not, the next `Task`s in the `TaskGraph`
        are released to the Simulator.

        Args:
            event (`Event`): The event containing the task that finished.
        """
        # Remove the Task from the WorkerPool and invoke it's finish method.
        task_placed_at_worker_pool = self._worker_pools.get_worker_pool(
            event.task.worker_pool_id
        )
        task_placed_at_worker_pool.remove_task(current_time=event.time, task=event.task)
        event.task.finish()

        # Log the TASK_FINISHED event into the CSV.
        self._finished_tasks += 1
        self._csv_logger.debug(
            f"{event.time.time},TASK_FINISHED,{event.task.name},{event.task.timestamp},"
            f"{event.task.task_graph},"
            f"{event.task.completion_time.to(EventTime.Unit.US).time},"
            f"{event.task.deadline.to(EventTime.Unit.US).time},{event.task.id}"
        )

        # If the TaskGraph corresponding to this task finished too, log that event.
        task_graph = self._workload.get_task_graph(event.task.task_graph)
        if task_graph is not None and task_graph.is_complete():
            self._finished_task_graphs += 1
            tardiness = (
                EventTime.zero()
                if task_graph.deadline > event.time
                else event.time - task_graph.deadline
            )
            self._csv_logger.debug(
                f"{event.time.time},TASK_GRAPH_FINISHED,{task_graph.name},"
                f"{task_graph.deadline.to(EventTime.Unit.US).time},"
                f"{tardiness.to(EventTime.Unit.US).time}"
            )
            if task_graph.deadline < event.time:
                self._missed_task_graph_deadlines += 1
            self._logger.info(
                "[%s] Finished the TaskGraph %s with a deadline %s at the "
                "completion of the task %s with a tardiness of %s.",
                event.time.to(EventTime.Unit.US).time,
                task_graph.name,
                task_graph.deadline,
                event.task.name,
                tardiness,
            )

        # Log if the task missed its deadline or not.
        if event.time > event.task.deadline:
            self._missed_task_deadlines += 1
            self._csv_logger.debug(
                f"{event.time.time},MISSED_DEADLINE,{event.task.name},"
                f"{event.task.timestamp},"
                f"{event.task.deadline.to(EventTime.Unit.US).time},{event.task.id}"
            )

        # Log if the TaskGraph missed its deadline.
        if task_graph is not None and event.time > task_graph.deadline:
            self._logger.warn(
                "[%s] Missed the deadline %s of the TaskGraph %s by %s.",
                event.time.to(EventTime.Unit.US).time,
                task_graph.deadline,
                task_graph.name,
                (event.time - task_graph.deadline).to(EventTime.Unit.US),
            )
            self._csv_logger.debug(
                f"{event.time.time},MISSED_TASK_GRAPH_DEADLINE,{task_graph.name},"
                f"{task_graph.deadline.to(EventTime.Unit.US).time}"
            )

        # The given task has finished execution, unlock dependencies from the `Workload`
        released_tasks, cancelled_tasks = self._workload.notify_task_completion(
            event.task, event.time
        )
        self._logger.info(
            "[%s] Notified the Workload of the completion of %s from %s, "
            "and received %s new tasks.",
            event.time.time,
            event.task,
            event.task.task_graph,
            len(released_tasks),
        )

        # The given TaskGraph has finished execution, unlock new TaskGraphs from the
        # `Workload`.
        if task_graph is not None and task_graph.is_complete():
            released_tasks_from_new_task_graph = (
                self._workload.notify_task_graph_completion(task_graph, event.time)
            )
            self._logger.info(
                "[%s] Notified the Workload of the completion of %s, "
                "and received %s new Tasks from new TaskGraphs.",
                event.time.time,
                task_graph.name,
                len(released_tasks_from_new_task_graph),
            )
            released_tasks.extend(released_tasks_from_new_task_graph)

        # Add events to cancel the required tasks.
        for index, task in enumerate(cancelled_tasks, start=1):
            event = Event(
                event_type=EventType.TASK_CANCEL, time=task.cancellation_time, task=task
            )
            self._event_queue.add_event(event)
            self._logger.info(
                "[%s] (%s/%s) Added %s for %s to the event queue.",
                event.time.to(EventTime.Unit.US).time,
                index,
                len(cancelled_tasks),
                event,
                task,
            )

        # Add events corresponding to the dependencies.
        for index, task in enumerate(released_tasks, start=1):
            event = Event(
                event_type=EventType.TASK_RELEASE,
                time=max(task.release_time, event.time),
                task=task,
            )
            self._event_queue.add_event(event)
            self._logger.info(
                "[%s] (%s/%s) Added %s for %s to the event queue.",
                event.time.time,
                index,
                len(released_tasks),
                event,
                task,
            )

    def __handle_task_preempt(self, event: Event) -> None:
        """Handles the TASK_PREEMPT event. The `Task` is removed from the `WorkerPool`
        where it was running and the `Task`'s preemption is logged into the CSV.

        Args:
            event (Event): The event to handle.
        """
        task = event.task
        self._csv_logger.debug(
            f"{event.time.time},TASK_PREEMPT,{task.name},{task.timestamp},{task.id}"
        )
        worker_pool = self._worker_pools.get_worker_pool(task.worker_pool_id)
        worker_pool.remove_task(current_time=event.time, task=task)
        task.preempt(event.time)

    def __handle_task_placement(self, event: Event, workload: Workload) -> None:
        """Handles the TASK_PLACEMENT event. The `Task` is placed on the `WorkerPool`
        where it was scheduled to run and the placement is moved further in the future
        if the `WorkerPool` is not ready to run the `Task`.

        Args:
            event (Event): The event to handle.
            workload (Workload): The `Workload` where the `TaskGraph`s for the `Task`
                are stored.
        """
        self._logger.debug(
            "[%s] Handling TASK_PLACEMENT event: %s.",
            event.time.to(EventTime.Unit.US).time,
            event,
        )
        task = event.task
        assert (
            task.id in self._future_placement_events
        ), "Inconsistency in future placements."
        task_graph = workload.get_task_graph(task.task_graph)
        assert task_graph is not None, "Inconsistency in Task placement and Workload."
        if not task.is_ready_to_run(task_graph):
            if task.state == TaskState.CANCELLED or task_graph.is_cancelled():
                # The Task was cancelled. Consume the event.
                self._logger.info(
                    "[%s] The Task %s was cancelled. Removing the event.",
                    event.time,
                    task,
                )
                del self._future_placement_events[task.id]
                if task.state != TaskState.CANCELLED:
                    # If the TaskGraph was cancelled, but the Task was not, then we
                    # emit a specific cancellation event for all the tasks from this
                    # task so that the status is correctly available to the schedulers.
                    for cancelled_task in task_graph.cancel(task, event.time):
                        self._event_queue.add_event(
                            Event(
                                event_type=EventType.TASK_CANCEL,
                                time=cancelled_task.cancellation_time,
                                task=cancelled_task,
                            )
                        )
                return
            else:
                # If the Task is not ready to run and wasn't cancelled,
                # find the next possible time to try executing the task.
                parent_completion_time = max(
                    parent.remaining_time for parent in task_graph.get_parents(task)
                )
                next_placement_time = event.time + max(
                    parent_completion_time, EventTime(1, EventTime.Unit.US)
                )
                next_placement_event = Event(
                    event_type=event.event_type,
                    time=next_placement_time,
                    task=event.task,
                    placement=event.placement,
                )
                self._future_placement_events[task.id] = next_placement_event
                self._event_queue.add_event(next_placement_event)
                self._logger.info(
                    "[%s] The Task %s was not ready to run, and has been pushed for "
                    "later placement at %s.",
                    event.time.to(EventTime.Unit.US).time,
                    task,
                    next_placement_time,
                )
                self._csv_logger.debug(
                    f"{event.time.time},TASK_NOT_READY,{task.name},{task.timestamp},"
                    f"{task.id},{event.placement.worker_pool_id}"
                )
                return
        # Initialize the task at the given placement time, and place it on
        # the WorkerPool.
        worker_pool = self._worker_pools.get_worker_pool(event.placement.worker_pool_id)
        assert (
            worker_pool is not None
        ), f"No WorkerPool found with ID: {event.placement.worker_pool_id}."
        success = worker_pool.place_task(
            task,
            execution_strategy=event.placement.execution_strategy,
            worker_id=event.placement.worker_id,
        )
        if success:
            task.start(event.time, variance=self._runtime_variance)
            resource_allocation_str = ",".join(
                [
                    ",".join((resource.name, resource.id, str(quantity)))
                    for resource, quantity in worker_pool.get_allocated_resources(task)
                ]
            )

            self._csv_logger.debug(
                f"{event.time.time},TASK_PLACEMENT,{task.name},{task.task_graph},"
                f"{task.timestamp},{task.id},{event.placement.worker_pool_id},"
                f"{event.placement.execution_strategy.runtime.time},"
                f"{resource_allocation_str}"
            )
            self._logger.info(
                "[%s] Placed %s on %s.", event.time.time, task, worker_pool
            )
            del self._future_placement_events[task.id]
        else:
            next_placement_time = event.time + EventTime(1, EventTime.Unit.US)
            next_placement_event = Event(
                event_type=event.event_type,
                time=next_placement_time,
                task=event.task,
                placement=event.placement,
            )
            self._event_queue.add_event(next_placement_event)
            self._future_placement_events[task.id] = next_placement_event
            self._logger.warning(
                "[%s] Task %s cannot be placed on worker %s, pushing placement to %s.",
                event.time.time,
                task,
                worker_pool,
                next_placement_time,
            )
            self._csv_logger.debug(
                f"{event.time.time},WORKER_NOT_READY,{task.name},{task.timestamp},"
                f"{task.id},{event.placement.worker_pool_id}"
            )

    def __handle_task_migration(self, event: Event) -> None:
        """Handles the TASK_MIGRATION event. This event must be followed by a
        `TASK_PREEMPT` event that preempts the `Task` from the `WorkerPool` where it
        was last placed. The `Task` is then placed on the `WorkerPool` where it is to
        be migrated.

        Args:
            event (Event): The event to handle.
        """
        task = event.task
        assert (
            task.state == TaskState.PREEMPTED
        ), f"The {task} was not PREEMPTED before being MIGRATED."

        last_preemption = task.last_preemption
        assert last_preemption is not None, f"The {task} did not have a preemption."
        self._logger.info(
            "[%s] Migrating %s from %s to %s.",
            event.time.time,
            task,
            last_preemption.old_worker_pool,
            event.placement,
        )
        self._logger.debug(
            "[%s] The resource requirements of %s were %s.",
            event.time.time,
            task,
            task.resource_requirements,
        )

        worker_pool = self._worker_pools.get_worker_pool(event.placement.worker_pool_id)
        success = worker_pool.place_task(task)
        if success:
            task.resume(event.time, worker_pool_id=event.placement.worker_pool_id)
            self._logger.debug(
                "[%s] The state of the WorkerPool(%s) is %s.",
                event.time.time,
                event.placement.worker_pool_id,
                worker_pool.resources,
            )
            resource_allocation_str = ",".join(
                [
                    ",".join((resource.name, resource.id, str(quantity)))
                    for resource, quantity in worker_pool.get_allocated_resources(task)
                ]
            )
            self._csv_logger.debug(
                f"{event.time.time},TASK_MIGRATED,{task.name},{task.timestamp},"
                f"{task.id},{last_preemption.old_worker_pool},"
                f"{event.placement.worker_pool_id},{resource_allocation_str}"
            )
        else:
            self._logger.warning(
                "[%s] Task %s cannot be migrated to worker %s.",
                event.time.time,
                task,
                worker_pool,
            )

    def __handle_task_graph_release(self, event: Event) -> None:
        """Handles an event of type `TASK_GRAPH_RELEASE`.

        Currently, this method just outputs the information about the TaskGraph into
        the CSV."""
        if event.event_type != EventType.TASK_GRAPH_RELEASE:
            raise ValueError(
                f"__handle_task_graph_release called with event of type {event.type}."
            )
        task_graph: TaskGraph = self._workload.get_task_graph(event.task_graph)
        if task_graph is None:
            raise ValueError(f"TaskGraph {event.task_graph} not found in the Workload.")
        self._csv_logger.info(
            "%s,TASK_GRAPH_RELEASE,%s,%s,%s,%s,%s",
            event.time.to(EventTime.Unit.US).time,
            task_graph.release_time.to(EventTime.Unit.US).time,
            task_graph.deadline.to(EventTime.Unit.US).time,
            task_graph.name,
            len(task_graph.get_nodes()),
            task_graph.critical_path_runtime.to(EventTime.Unit.US).time,
        )
        if self._log_task_graphs:
            # Log a DOT representation of the TaskGraph, if requested.
            task_graph.to_dot(os.path.join(self._log_dir, f"{task_graph.name}.dot"))

    def __handle_update_workload(self, event: Event) -> None:
        """Handles an Event of type `UPDATE_WORKLOAD`.

        This method reaches out to the WorkloadLoader to load the next batch of
        TaskGraphs into the Workload. It then releases the releasable Tasks from the
        Workload into the Simulator."""
        if event.event_type != EventType.UPDATE_WORKLOAD:
            raise ValueError(
                f"__handle_update_workload called with event of type {event.type}."
            )
        updated_workload = self._workload_loader.get_next_workload(
            current_time=self._simulator_time
        )
        if updated_workload is None:
            self._logger.info(
                "[%s] The WorkloadLoader %s has no more updates to release "
                "into the Workload.",
                self._simulator_time.to(EventTime.Unit.US).time,
                type(self._workload_loader).__name__,
            )
            self._csv_logger.info("%s,UPDATE_WORKLOAD,0,0", self._simulator_time.time)
        else:
            self._workload = updated_workload
            # Notify the Scheduler of the updated Workload.
            self._scheduler.notify_workload_updated(
                sim_time=self._simulator_time,
                workload=self._workload,
                worker_pools=self._worker_pools,
            )

            # Release the Tasks that have become available.
            releasable_tasks = self._workload.get_releasable_tasks()
            self._logger.info(
                "[%s] The WorkloadLoader %s has %s TaskGraphs that released %s tasks.",
                self._simulator_time.to(EventTime.Unit.US).time,
                type(self._workload_loader).__name__,
                len(self._workload.task_graphs),
                len(releasable_tasks),
            )
            self._csv_logger.info(
                "%s,UPDATE_WORKLOAD,%s,%s",
                event.time.to(EventTime.Unit.US).time,
                len(self._workload.task_graphs),
                len(releasable_tasks),
            )

            # # Add the TaskGraphRelease events into the system.
            # for task_graph_name, task_graph in self._workload.task_graphs.items():
            #     event = Event(
            #         event_type=EventType.TASK_GRAPH_RELEASE,
            #         time=task_graph.release_time,
            #         task_graph=task_graph_name,
            #     )
            #     self._event_queue.add_event(event)
            #     self._logger.info(
            #         "[%s] Added %s to the event queue.",
            #         self._simulator_time.to(EventTime.Unit.US).time,
            #         event,
            #     )

            max_release_time = self._simulator_time
            for task in releasable_tasks:
                event = Event(
                    event_type=EventType.TASK_RELEASE, time=task.release_time, task=task
                )
                self._event_queue.add_event(event)
                self._logger.info(
                    "[%s] Added %s for %s from %s to the event queue.",
                    self._simulator_time.time,
                    event,
                    task,
                    task.task_graph,
                )
                if task.release_time > max_release_time:
                    max_release_time = task.release_time

            next_update_event = Event(
                event_type=EventType.UPDATE_WORKLOAD,
                time=(
                    max_release_time + EventTime(1, EventTime.Unit.US)
                    if self._workload_update_interval.is_invalid()
                    else self._simulator_time + self._workload_update_interval
                ),
            )
            self._event_queue.add_event(next_update_event)
            self._logger.info(
                "[%s] Added %s to the event queue.",
                self._simulator_time.time,
                next_update_event,
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
        self._logger.info(
            "[%s] Received %s from the event queue.", self._simulator_time.time, event
        )

        if event.event_type == EventType.SIMULATOR_START:
            # Start of the simulator loop.
            self._csv_logger.debug(f"{event.time.time},SIMULATOR_START")
            self._logger.info(
                "[%s] Starting the simulator loop.",
                event.time.to(EventTime.Unit.US).time,
            )
            # placements = self._scheduler.start(
            #     event.time, self._workload.work_profiles, self._worker_pools
            # )
            # for placement in placements:
            #     if (
            #         placement.placement_type
            #         != Placement.PlacementType.LOAD_WORK_PROFILE
            #     ):
            #         raise RuntimeError(
            #             f"A Placement of type {placement.placement_type} was "
            #             f"returned by the scheduler. Only "
            #             f"{Placement.PlacementType.LOAD_WORK_PROFILE} is supported."
            #         )
            #     profile_load_event = Event(
            #         event_type=EventType.LOAD_PROFILE,
            #         time=placement.placement_time,
            #         placement=placement,
            #     )
            #     self._logger.debug(
            #         "[%s] Adding %s to the event queue as part of scheduler start.",
            #         event.time.time,
            #         profile_load_event,
            #     )
            #     self._event_queue.add_event(profile_load_event)
        elif event.event_type == EventType.TASK_CANCEL:
            self.__handle_task_cancellation(event)
        elif event.event_type == EventType.EVICT_PROFILE:
            self.__handle_profile_eviction(event)
        elif event.event_type == EventType.TASK_FINISHED:
            self.__handle_task_finished(event)
        elif event.event_type == EventType.TASK_GRAPH_RELEASE:
            self.__handle_task_graph_release(event)
        elif event.event_type == EventType.TASK_RELEASE:
            self.__handle_task_release(event)
        elif event.event_type == EventType.UPDATE_WORKLOAD:
            self.__handle_update_workload(event)
        elif event.event_type == EventType.TASK_PREEMPT:
            self.__handle_task_preempt(event)
        elif event.event_type == EventType.TASK_MIGRATION:
            self.__handle_task_migration(event)
        elif event.event_type == EventType.LOAD_PROFILE:
            self.__handle_profile_loading(event)
        elif event.event_type == EventType.TASK_PLACEMENT:
            self.__handle_task_placement(event, self._workload)
        elif event.event_type == EventType.SCHEDULER_START:
            self.__handle_scheduler_start(event)
        elif event.event_type == EventType.SCHEDULER_FINISHED:
            self.__handle_scheduler_finish(event)
        elif event.event_type == EventType.SIMULATOR_END:
            # End of the simulator loop.
            self._csv_logger.debug(
                f"{event.time.time},SIMULATOR_END,{self._finished_tasks},"
                f"{self._cancelled_tasks},{self._missed_task_deadlines},"
                f"{self._finished_task_graphs},"
                f"{len(self._workload.get_cancelled_task_graphs())},"
                f"{self._missed_task_graph_deadlines}"
            )
            self._logger.info("[%s] Ending the simulator loop.", event.time.time)
            return True
        elif event.event_type == EventType.LOG_UTILIZATION:
            self.__log_utilization(event.time)
        else:
            raise ValueError(f"[{event.time}] Retrieved event of unknown type: {event}")
        return False

    def __step(self, step_size: EventTime = EventTime(1, EventTime.Unit.US)) -> None:
        """Advances the clock by the given `step_size`.

        Args:
            step_size (`EventTime`) [default=1]: The amount by which to advance
                the clock (in us).
        """
        if step_size < EventTime.zero():
            raise ValueError(f"Simulator cannot step backwards {step_size}")

        # Step the simulator for the required steps and construct TASK_FINISHED events
        # for any tasks that were able to complete their execution.
        self._logger.info(
            "[%s] Stepping for %s timesteps.",
            self._simulator_time.time,
            step_size,
        )
        task_finished_events = []
        for worker_pool in self._worker_pools.worker_pools:
            for task in worker_pool.step(self._simulator_time, step_size):
                task_finished_event = Event(
                    event_type=EventType.TASK_FINISHED,
                    time=self._simulator_time + step_size,
                    task=task,
                )
                task_finished_events.append(task_finished_event)

        # Update the simulator time, and add the TASK_FINISHED events to the queue for
        # further processing.
        self._simulator_time += step_size
        self._logger.debug(
            "[%s] The stepping yielded the following completed tasks: %s.",
            self._simulator_time.time,
            [event.task.unique_name for event in task_finished_events],
        )
        for task_finished_event in task_finished_events:
            self._event_queue.add_event(task_finished_event)
            self._logger.info(
                "[%s] Added %s to the event queue.",
                self._simulator_time.time,
                task_finished_event,
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
        if scheduler_frequency < EventTime.zero():
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
                    "[%s] The scheduler invocations are late. Supposed to start at %s.",
                    event.time.time,
                    next_scheduler_time,
                )
                scheduler_start_time = event.time + EventTime(1, EventTime.Unit.US)
            else:
                scheduler_start_time = next_scheduler_time
        self._logger.debug(
            "[%s] Executing the next scheduler at %s because the frequency was %s.",
            event.time.time,
            scheduler_start_time,
            scheduler_frequency,
        )

        # End the loop according to the timeout, if reached.
        if scheduler_start_time >= loop_timeout:
            self._logger.info(
                "[%s] The next scheduler start was scheduled at %s, but the loop "
                "timeout is %s. Ending the loop.",
                event.time.time,
                scheduler_start_time,
                loop_timeout,
            )
            return Event(event_type=EventType.SIMULATOR_END, time=loop_timeout)

        # Find sources of existing or ongoing work in the Simulator.
        # Find the minimum remaining time from all the running / scheduled tasks.
        running_tasks: Sequence[Task] = self._worker_pools.get_placed_tasks() + list(
            [placement.task for placement in self._future_placement_events.values()]
        )
        remaining_times = []
        for task in running_tasks:
            if task.state == TaskState.SCHEDULED:
                remaining_times.append(
                    (
                        task.unique_name,
                        task.expected_start_time,
                        task.expected_start_time + task.remaining_time,
                    )
                )
            elif task.state == TaskState.RUNNING:
                remaining_times.append(
                    (
                        task.unique_name,
                        self._simulator_time,
                        self._simulator_time + task.remaining_time,
                    )
                )
            else:
                self._logger.warn(
                    f"The task {task.unique_name} was in an "
                    f"unexpected state: {task.state}."
                )
        self._logger.debug(
            "[%d] The running tasks along with their start time "
            "and expected completion times are: %s.",
            event.time.to(EventTime.Unit.US).time,
            [
                f"{task_name} ({start_time}, {completion_time})"
                for task_name, start_time, completion_time in remaining_times
            ],
        )
        minimum_running_task_completion_time = (
            min(
                map(itemgetter(2), remaining_times),
                default=EventTime(sys.maxsize, EventTime.Unit.US),
            )
            + self._scheduler_delay
        )

        # Get the schedulable tasks that are waiting to be executed.
        schedulable_tasks = self._workload.get_schedulable_tasks(
            time=event.time,
            lookahead=self._scheduler.lookahead,
            preemption=self._scheduler.preemptive,
            retract_schedules=self._scheduler.retract_schedules,
            worker_pools=self._worker_pools,
            policy=self._scheduler.policy,
            branch_prediction_accuracy=self._scheduler.branch_prediction_accuracy,
            release_taskgraphs=self._scheduler.release_taskgraphs,
        )
        self._logger.debug(
            "[%s] The schedulable tasks are %s.",
            event.time.time,
            [task.unique_name for task in schedulable_tasks],
        )
        next_task_release_event = self._event_queue.get_next_event_of_type(
            EventType.TASK_RELEASE
        )
        next_workload_update_event = self._event_queue.get_next_event_of_type(
            EventType.UPDATE_WORKLOAD
        )
        next_event = self._event_queue.peek()
        self._logger.debug(
            "[%s] The next event in the queue is %s, the next TASK_RELEASE event is "
            "%s, and the next UPDATE_WORKLOAD event is %s.",
            event.time.time,
            next_event,
            next_task_release_event,
            next_workload_update_event,
        )

        # If there is either existing work in the form of events in the queue or tasks
        # waiting to be scheduled, or currently running tasks that can lead to more
        # work, adjust the scheduler invocation time accordingly, or end the loop.
        if (
            next_event is None
            and len(schedulable_tasks) == 0
            and len(running_tasks) == 0
        ):
            self._logger.info(
                "[%s] There are no currently schedulable tasks, no running tasks, "
                "and no events available in the event queue. Ending the loop.",
                event.time.time,
            )
            return Event(
                event_type=EventType.SIMULATOR_END,
                time=event.time + EventTime(1, EventTime.Unit.US),
            )
        elif len(running_tasks) > 0 and self._run_scheduler_at_worker_free:
            # The scheduler was requested to be invoked at the completion of the next
            # task event. We move the scheduler event to the time when the earliest
            # task ends.
            self._logger.debug(
                "[%s] The scheduler was requested to run when any Worker becomes free. "
                "The minimum running task completion time is %s, so we invoke the "
                "scheduler at that time.",
                event.time.to(EventTime.Unit.US).time,
                minimum_running_task_completion_time,
            )
            scheduler_start_time = minimum_running_task_completion_time + EventTime(
                1, EventTime.Unit.US
            )
        elif (
            len(schedulable_tasks) == 0
            or all(
                task.state in (TaskState.RUNNING, TaskState.SCHEDULED)
                for task in schedulable_tasks
            )
            or self._worker_pools.is_full()
            or all(
                len(
                    worker.get_compatible_strategies(
                        schedulable_task.available_execution_strategies
                    )
                )
                == 0
                for schedulable_task in schedulable_tasks
                for worker_pool in self._worker_pools.worker_pools
                for worker in worker_pool.workers
                if worker.is_available(schedulable_task.profile) == EventTime.zero()
            )
        ):
            # If there are no schedulable tasks currently, or all schedulable tasks are
            # already running (in a preemptive scheduling scenario), or the WorkerPool
            # is full, adjust the scheduler invocation time according to either the
            # time of invocation of the next event, or the minimum completion time
            # of a running task.

            next_task_release_event_invocation_time = (
                next_task_release_event.time.to(EventTime.Unit.US)
                + self._scheduler_delay.to(EventTime.Unit.US)
                if next_task_release_event is not None
                else EventTime(sys.maxsize, EventTime.Unit.US)
            )
            next_workload_update_event_time = (
                next_workload_update_event.time.to(EventTime.Unit.US)
                if next_workload_update_event is not None
                else EventTime(sys.maxsize, EventTime.Unit.US)
            )

            next_event_time = min(
                minimum_running_task_completion_time,
                next_task_release_event_invocation_time,
                next_workload_update_event_time,
            )
            self._logger.debug(
                "[%s] The next event time was %s because the minimum running task "
                "completion time was %s, the next TASK_RELEASE event in the queue "
                "was at %s, and the next workload update event was at %s.",
                event.time.to(EventTime.Unit.US).time,
                next_event_time,
                minimum_running_task_completion_time,
                next_task_release_event_invocation_time,
                next_workload_update_event_time,
            )

            adjusted_scheduler_start_time = max(scheduler_start_time, next_event_time)

            if scheduler_start_time != adjusted_scheduler_start_time:
                if adjusted_scheduler_start_time >= loop_timeout:
                    self._logger.info(
                        "[%s] The adjusted scheduler start was scheduled at %s, but "
                        "the loop timeout is %s. Ending the loop.",
                        event.time.time,
                        adjusted_scheduler_start_time,
                        loop_timeout,
                    )
                    return Event(event_type=EventType.SIMULATOR_END, time=loop_timeout)

                self._logger.info(
                    "[%s] The scheduler start time was pushed from %s to %s since "
                    "either the next running task finishes at %s or the next "
                    "TASK_RELEASE event is being invoked at %s.",
                    event.time.to(EventTime.Unit.US).time,
                    scheduler_start_time,
                    adjusted_scheduler_start_time,
                    minimum_running_task_completion_time,
                    next_task_release_event_invocation_time,
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

        # Run the scheduler.
        placements = self._scheduler.schedule(
            event.time,
            self._workload,
            self._worker_pools,
        )
        if placements is None:
            raise ValueError(
                f"Received no Placements object from the Scheduler at {event.time}.",
            )

        # Calculate the time at which the placements need to be applied.
        placement_time = event.time + placements.runtime

        # Save the placements until the placement time arrives.
        self._last_scheduler_placements = placements

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
        for worker_pool in self._worker_pools.worker_pools:
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
