import heapq
from enum import Enum
from typing import Type, Sequence, Optional

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
    """
    def __init__(self,
                 worker_pools: Sequence[WorkerPool],
                 scheduler: Type[BaseScheduler],
                 job_graph: JobGraph,
                 loop_timeout: float = float('inf'),
                 scheduler_frequency: float = -1.0,
                 ):
        self._worker_pools = {worker_pool.id: worker_pool for worker_pool in
                              worker_pools}
        self._scheduler = scheduler
        self._job_graph = job_graph
        self._simulator_time = 0
        self._scheduler_frequency = scheduler_frequency

        # Internal data.
        self._last_scheduler_start_time = 0
        self._available_tasks = []

        # Initialize the event queue, and add a SIMULATOR_START / SIMULATOR_END
        # task to signify the beginning and completion of the simulator loop.
        self._event_queue = EventQueue()
        self._event_queue.add_event(Event(event_type=EventType.SIMULATOR_START,
                                          time=0, task=None))
        self._event_queue.add_event(Event(event_type=EventType.SIMULATOR_END,
                                          time=loop_timeout, task=None))

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
        # At the beginning, this should consist of all the sensor tasks that
        # we expect to run during the execution of the workload, along with
        # their expected release times.
        for task in task_graph.get_all_released_tasks():
            self._event_queue.add_event(
                    Event(event_type=EventType.TASK_RELEASE,
                          time=task.release_time, task=task)
                    )

        # Run the simulator loop.
        while True:
            event = self._event_queue.next()

            # Advance the clock until the occurrence of this event.
            self.__step(step_size=self._simulator_time - event.time)

            if event.event_type == EventType.SIMULATOR_START:
                # Start of the simulator loop.
                print("[x] Starting the simulator loop at time {}.".format(
                      event.time))
            elif event.event_type == EventType.SIMULATOR_END:
                # End of the simulator loop.
                print("[x] Ending the simulator loop at time {}".format(
                      event.time))
                break
            elif event.event_type == EventType.TASK_RELEASE:
                # Release a task for the scheduler.
                self._available_tasks.append(event.task)
            elif event.event_type == EventType.TASK_FINISHED:
                # The given task has finished execution, unlock dependencies.
                new_tasks = task_graph.notify_task_completion(event.task)

                # Add events corresponding to the dependencies.
                for task in new_tasks:
                    self._event_queue.add_event(
                            Event(event_type=EventType.TASK_RELEASE,
                                  time=task.release_time, task=task))
            elif event.event_type == EventType.SCHEDULER_START:
                # Execute the scheduler, and insert an event notifying the
                # end of the scheduler into the loop.
                self._last_scheduler_start_time = event.time
                print("[x] Running the scheduler at time {}".format(
                      self._simulator_time))
                self._event_queue.add_event(self.__run_scheduler(event))
            elif event.event_type == EventType.SCHEDULER_FINISHED:
                # The scheduler has finished its execution, insert an event
                # for the next invocation of the scheduler.
                self._event_queue.add_event(
                        self.__get_next_scheduler_event(event))
            else:
                print("[x] Retrieved event of unknown type: {}".format(
                      event.event_type))

    def __step(self, step_size: float = 1.0):
        """Advances the clock by the given `step_size`.

        Args:
            step_size (`float`) [default=1.0]: The amount by which to advance
                the clock.
        """
        for worker_pool in self._worker_pools:
            worker_pool.step(self._simulator_time, step_size)
        self._simulator_time += step_size

    def __get_next_scheduler_event(self, event: Event) -> Event:
        """Computes the next event when the scheduler should run.

        Args:
            event (`Event`): The event at which the last scheduler invocation
                finished.

        Returns:
            An event signifying when the next scheduler invocation should be.
        """
        scheduler_start_time = None
        if self._scheduler_frequency < 0:
            # Insert a new scheduler event for the next step.
            scheduler_start_time = event.time + 1.0
        else:
            # Calculate when the next scheduler event was going to
            # occur according to its periodicity.
            next_scheduler_time = (self._last_scheduler_start_time +
                                   self._scheduler_frequency)

            # If that time has already occurred, invoke a scheduler
            # in the next time step, otherwise wait until that time.
            if next_scheduler_time < event.time:
                print("[x] The scheduler invocations are running late.\
                        Supposed to start at {}, currently {}".format(
                            next_scheduler_time, event.time))
                scheduler_start_time = event.time + 1.0
            else:
                scheduler_start_time = next_scheduler_time
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
        """
        scheduler_runtime, task_placement = self._scheduler.schedule(
                self._available_tasks, task_graph, self._worker_pools)
        placement_time = event.time + scheduler_runtime

        # Place the task on the assigned worker pool, and reset the available
        # events to the tasks that could not be placed.
        # TODO (Sukrit): Should these tasks be moved to a PAUSED state?
        available_tasks = []
        for task, placement in task_placement:
            if placement is None:
                available_tasks.append(task)
            else:
                worker_pool = self._worker_pools[placement]
                # Initialize the task at the given placement time, and place
                # it on the WorkerPool.
                task.start(placement_time)
                worker_pool.place_task(task)
        self._available_tasks = available_tasks

        return Event(event_type=EventType.SCHEDULER_FINISHED,
                     time=placement_time)
