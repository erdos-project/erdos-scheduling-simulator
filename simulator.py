import heapq
from enum import Enum
from typing import Type, Sequence, Optional

import absl

import utils
from workers import WorkerPool
from workload import JobGraph, TaskGraph, Task
from schedulers import BaseScheduler


class EventType(Enum):
    """Represents the different Events that a simulator has to simulate."""
    SCHEDULER_START = 0     # Requires the simulator to invoke the scheduler.
    SCHEDULER_FINISHED = 1  # Signifies the end of the scheduler loop.
    TASK_RELEASE = 2        # Ask the simulator to release the task.
    TASK_FINISHED = 3       # Notify the simulator of the end of a task.
    SIMULATOR_START = 4     # Signify the start of the simulator loop.
    SIMULATOR_END = 5       # Signify the end of the simulator loop.


class Event(object):
    """Represents an `Event` that is managed by the `EventQueue` and informs
    what action the simulator needs to do at the given simulator time.

    Args:
        event_type (`EventType`): The type of the event.
        time (`float`): The simulator time at which the event occurred.
        task (`Optional[Task]`): The task associated with this event if it is
            of type `TASK_RELEASE` or `TASK_FINISHED`.

    Raises:
        `ValueError` if the event is of type `TASK_RELEASE` or `TASK_FINISHED`
        and no associated task is provided.
    """
    def __init__(self, event_type: EventType, time: float,
                 task: Optional[Task] = None):
        if (event_type == EventType.TASK_RELEASE or
                event_type == EventType.TASK_FINISHED):
            if task is None:
                raise ValueError("No associated task provided with {}".
                                 format(event_type))
        self._event_type = event_type
        self._time = time
        self._task = task

    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        return "Event(time={}, type={})".format(self.time, self.event_type)

    def __repr__(self):
        return str(self)

    @property
    def time(self) -> float:
        return self._time

    @property
    def event_type(self) -> EventType:
        return self._event_type

    @property
    def task(self) -> Optional[Task]:
        return self._task


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
        job_graph (`JobGraph`): A static directed graph that represents the
            known structure of the computation.
        loop_timeout (`float`) [default=float('inf')]: The simulator time upto
            which to run the loop. The default runs until we have exhausted all
            the events in the system.
        scheduler_frequency (`float`) [default=-1]: The time between two
            subsequent scheduler invocations. The default invokes a new run of
            the scheduler just after the previous one has completed.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """
    def __init__(self,
                 worker_pools: Sequence[WorkerPool],
                 scheduler: Type[BaseScheduler],
                 job_graph: JobGraph,
                 loop_timeout: float = float('inf'),
                 scheduler_frequency: float = -1.0,
                 _flags: Optional['absl.flags'] = None,
                 ):
        if not isinstance(scheduler, BaseScheduler):
            raise ValueError(
                    "Scheduler must implement the BaseScheduler interface.")
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
            self._csv_logger = utils.setup_csv_logging(
                                    name=self.__class__.__name__,
                                    log_file=_flags.csv_file_name)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)
            self._csv_logger = utils.setup_csv_logging(
                                    name=self.__class__.__name__,
                                    log_file=None)

        self._worker_pools = {worker_pool.id: worker_pool for worker_pool in
                              worker_pools}
        self._scheduler = scheduler
        self._job_graph = job_graph
        self._simulator_time = 0
        self._scheduler_frequency = scheduler_frequency
        self._loop_timeout = loop_timeout

        # Internal data.
        self._last_scheduler_start_time = 0
        self._last_task_placement = []
        self._released_tasks = []

        # Initialize the event queue, and add a SIMULATOR_START / SIMULATOR_END
        # task to signify the beginning and completion of the simulator loop.
        # Also add a SCHEDULER_START event to invoke the scheduling loop.
        self._event_queue = EventQueue()

        sim_start_event = Event(event_type=EventType.SIMULATOR_START, time=0)
        self._event_queue.add_event(sim_start_event)
        self._logger.debug("Added {} to the event queue.".
                           format(sim_start_event))

        sched_start_event = Event(event_type=EventType.SCHEDULER_START, time=0)
        self._event_queue.add_event(sched_start_event)
        self._logger.debug("Added {} to the event queue.".
                           format(sched_start_event))

    def simulate(self, task_graph: TaskGraph):
        """Run the simulator loop.

        This loop requires an actual runtime instantiation of the `JobGraph`
        using the `TaskGraph`.

        Args:
            task_graph (`TaskGraph`): A graph of tasks that are currently
                available to execute, along with potential future tasks that
                could be released.
        """
        # Retrieve the set of released tasks from the graph.
        # At the beginning, this should consist of all the sensor tasks
        # that we expect to run during the execution of the workload, 
        # along with their expected release times.
        for task in task_graph.release_tasks():
            event = Event(event_type=EventType.TASK_RELEASE,
                          time=task.release_time, task=task)
            self._event_queue.add_event(event)
            self._logger.debug("Added {} for {} to the event queue.".
                               format(event, task))

        # Run the simulator loop.
        while True:
            if self.__handle_event(self._event_queue.next(), task_graph):
                break

    def __handle_event(self, event: Event, task_graph: TaskGraph) -> bool:
        """ Handles the next event from the EventQueue.

        Invoked by the simulator loop, and tested using unit tests.

        Args:
            event (`Event`): The event to handle.
            task_graph (`TaskGraph`): A graph of tasks that are currently
                available to execute, along with potential future tasks that
                could be released.

        Returns:
            `True` if the event is a SIMULATOR_END and the simulator loop
            should be stopped, `False` otherwise.
        """
        self._logger.debug("Received {} from the event queue.".format(event))
        # Advance the clock until the occurrence of this event.
        self.__step(step_size=self._simulator_time - event.time)

        if event.event_type == EventType.SIMULATOR_START:
            # Start of the simulator loop.
            self._logger.info("Starting the simulator loop at time {}".
                              format(event.time))
        elif event.event_type == EventType.SIMULATOR_END:
            # End of the simulator loop.
            self._logger.info("Ending the simulator loop at time {}".
                              format(event.time))
            return True
        elif event.event_type == EventType.TASK_RELEASE:
            # Release a task for the scheduler.
            self._released_tasks.append(event.task)
            self._logger.debug("Added {} to the available tasks.".
                               format(event))
            self._csv_logger.debug(
                "{sim_time},TASK_RELEASE,{task_name}_{timestamp},{task_id}".
                format(
                       sim_time=event.time,
                       task_name=event.task.name,
                       timestamp=event.task.timestamp,
                       task_id=event.task.id))
        elif event.event_type == EventType.TASK_FINISHED:
            print("Finished execution of Task for {} with timestamp: {}".
                  format(event.task.name, event.task.timestamp))
            self._csv_logger.debug(
                "{sim_time},TASK_FINISHED,{task_name}_{timestamp},{task_id}".
                format(
                       sim_time=event.time,
                       task_name=event.task.name,
                       timestamp=event.task.timestamp,
                       task_id=event.task.id))

            # The given task has finished execution, unlock dependencies.
            new_tasks = task_graph.notify_task_completion(event.task,
                                                          event.time)
            self._logger.debug("Notified the task graph of the completion of "
                               "{}, and received {} new tasks.".format(
                                   event.task, len(new_tasks)))

            # Add events corresponding to the dependencies.
            for index, task in enumerate(new_tasks, start=1):
                event = Event(event_type=EventType.TASK_RELEASE,
                              time=task.release_time, task=task)
                self._event_queue.add_event(event)
                self._logger.debug("({}/{}) Added {} for {} to the "
                                   "event queue.".format(index, len(new_tasks),
                                                         event, task))
        elif event.event_type == EventType.SCHEDULER_START:
            # Log the required CSV information.
            currently_placed_tasks = []
            for worker_pool in self._worker_pools.values():
                currently_placed_tasks.extend(worker_pool.get_placed_tasks())
            self._csv_logger.debug(
                "{sim_time},SCHEDULER_START,{released_tasks},{placed_tasks}".
                format(
                       sim_time=event.time,
                       released_tasks=len(self._released_tasks),
                       placed_tasks=len(currently_placed_tasks)))

            # Execute the scheduler, and insert an event notifying the
            # end of the scheduler into the loop.
            self._last_scheduler_start_time = event.time
            self._logger.info("Running the scheduler at time {}".
                              format(self._simulator_time))
            sched_finished_event = self.__run_scheduler(event, task_graph)
            self._event_queue.add_event(sched_finished_event)
            self._logger.debug("Added {} to the event queue.".
                               format(sched_finished_event))
        elif event.event_type == EventType.SCHEDULER_FINISHED:
            # Place the task on the assigned worker pool, and reset the
            # available events to the tasks that could not be placed.
            # TODO (Sukrit): Should these tasks be moved to a PAUSED state?
            self._logger.debug("Finished executing the scheduler initiated "
                               "at {}. Placing tasks.".format(
                                    self._last_scheduler_start_time))
            released_tasks = []
            num_placed_tasks = 0
            for task, placement in self._last_task_placement:
                if placement is None:
                    released_tasks.append(task)
                    self._logger.debug("Failed to place {}".format(task))
                else:
                    worker_pool = self._worker_pools[placement]
                    # Initialize the task at the given placement time, and
                    # place it on the WorkerPool.
                    task.start(event.time)
                    worker_pool.place_task(task)
                    self._logger.debug("Placed {} on {}".format(task,
                                                                worker_pool))
                    num_placed_tasks += 1

            # Log the required CSV information.
            self._csv_logger.debug(
                "{sim_time},SCHEDULER_FINISHED,{runtime},{placed_tasks}".
                format(
                       sim_time=event.time,
                       runtime=event.time - self._last_scheduler_start_time,
                       placed_tasks=num_placed_tasks))

            # Reset the available tasks and the last task placement.
            self._released_tasks = released_tasks
            self._last_task_placement = []

            # The scheduler has finished its execution, insert an event
            # for the next invocation of the scheduler.
            next_sched_event = self.__get_next_scheduler_event(
                        event, self._scheduler_frequency,
                        self._last_scheduler_start_time,
                        self._loop_timeout,
                    )
            self._event_queue.add_event(next_sched_event)
            self._logger.debug("Added {} to the event queue.".
                               format(next_sched_event))
        else:
            self._logger.error("Retrieved event of unknown type: {}".
                               format(event))
        return False

    def __step(self, step_size: float = 1.0):
        """Advances the clock by the given `step_size`.

        Args:
            step_size (`float`) [default=1.0]: The amount by which to advance
                the clock.
        """
        completed_tasks = []
        for worker_pool in self._worker_pools.values():
            completed_tasks.extend(worker_pool.step(self._simulator_time,
                                                    step_size))

        # Add TASK_FINISHED events for all the completed tasks.
        self._simulator_time += step_size
        for task in completed_tasks:
            task_finished_event = Event(event_type=EventType.TASK_FINISHED,
                                        time=self._simulator_time, task=task)
            self._event_queue.add_event(task_finished_event)
            self._logger.debug("Added {} for the task {} to the event queue.".
                               format(task_finished_event, task))

    def __get_next_scheduler_event(self,
                                   event: Event,
                                   scheduler_frequency: float,
                                   last_scheduler_start_time: float,
                                   loop_timeout: float = float('inf'),
                                   ) -> Event:
        """Computes the next event when the scheduler should run.

        This method returns a SIMULATOR_END event if either the loop timeout
        is reached, or there are no future task releases in the event queue
        or released tasks.

        Args:
            event (`Event`): The event at which the last scheduler invocation
                finished.
            scheduler_frequency (`float`): The frequency at which the simulator
                needs to be invoked.
            last_scheduler_start_time (`float`): The time at which the last
                invocation of scheduler occurred.

        Returns:
            An event signifying when the next scheduler invocation should be.
            May be of type SCHEDULER_START or SIMULATOR_END

        Raises:
            `ValueError` if an event type != SCHEDULER_FINISHED is passed.
        """
        if not (event.event_type == EventType.SCHEDULER_FINISHED):
            raise ValueError("Incorrect event type passed.")

        scheduler_start_time = None
        if scheduler_frequency < 0:
            # Insert a new scheduler event for the next step.
            scheduler_start_time = event.time + 1.0
        else:
            # Calculate when the next scheduler event was going to
            # occur according to its periodicity.
            next_scheduler_time = (last_scheduler_start_time +
                                   scheduler_frequency)

            # If that time has already occurred, invoke a scheduler
            # in the next time step, otherwise wait until that time.
            if next_scheduler_time < event.time:
                self._logger.warning("The scheduler invocations are late. "
                                     "Supposed to start at {}, currently {}".
                                     format(next_scheduler_time, event.time))
                scheduler_start_time = event.time + 1.0
            else:
                scheduler_start_time = next_scheduler_time

        # End the loop according to the timeout, if reached.
        if scheduler_start_time >= loop_timeout:
            return Event(event_type=EventType.SIMULATOR_END, time=loop_timeout)

        # If no events in the queue and no released tasks, end the loop.
        if len(self._released_tasks) == 0 and len(self._event_queue) == 0:
            return Event(event_type=EventType.SIMULATOR_END, time=loop_timeout)

        return Event(event_type=EventType.SCHEDULER_START,
                     time=scheduler_start_time)

    def __run_scheduler(self, event: Event, task_graph: TaskGraph) -> Event:
        """Run the scheduler.

        Args:
            event (`Event`): The event at which the scheduler was invoked.
            task_graph (`TaskGraph`): A graph of tasks that are currently
                available to execute, along with potential future tasks that
                could be released.

        Returns:
            An `Event` signifying the end of the scheduler invocation.

        Raises:
            `ValueError` if an event type != SCHEDULER_START is passed.
        """
        if not (event.event_type == EventType.SCHEDULER_START):
            raise ValueError("Incorrect event type passed.")
        scheduler_runtime, task_placement = self._scheduler.schedule(
                event.time, self._released_tasks, task_graph,
                self._worker_pools)
        placement_time = event.time + scheduler_runtime
        self._last_task_placement = task_placement

        return Event(event_type=EventType.SCHEDULER_FINISHED,
                     time=placement_time)
