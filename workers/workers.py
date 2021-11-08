import uuid
from workload import Resources, Task
from typing import Optional, Sequence, Type
from scheduler import BaseScheduler


class Worker(object):
    """A `Worker` is a virtual abstraction over a single machine.

    A `Worker` "owns" a certain set of `Resource`s, and contains a set of
    runtime threads that together execute a set of `Task`s assigned to it.

    Args:
        name (`str`): A name assigned to the particular instance of the Worker.
        resources (`Resource`): The set of `Resource`s owned by this worker.
        id (`UUID`): The ID of this particular Worker.
        num_threads (`int`): The number of threads in this Worker.
    """
    def __init__(self, name: str, resources: Resources, num_threads: int):
        self._name = name
        self._id = uuid.uuid4()
        self._resources = resources
        self._num_threads = num_threads
        self._placed_tasks = {}  # Tasks along with their execution status.

    def place_task(self, task: Task):
        """Places the task on this `Worker`.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.
        """
        raise NotImplementedError("Cannot place tasks right now.")

    def can_accomodate_task(self, task: Task) -> bool:
        """Checks if this `Worker` can accomodate the given `Task` based on
        its resource availability.

        Args:
            task (`Task`): The task to be placed on this `Worker`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        raise NotImplementedError("Cannot place tasks right now.")

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task` that is currently placed on this `Worker`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        placed_tasks = []
        for task, _ in self._placed_tasks:
            placed_tasks.append(task)
        return placed_tasks

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @property
    def resources(self):
        return self._resources

    @property
    def num_threads(self):
        return self._num_threads

    def __str__(self):
        return "Worker(name={}, id={}, resources={}, num_threads={})".format(
                self.name, self.id, self.resources, self.num_threads)

    def __repr__(self):
        return str(self)


class WorkerPool(object):
    """A `WorkerPool` acts as a collection of `Worker`s, and may perform its
    own scheduling (for a two-level scheduling architecture) across its
    workers.

    A Scheduler only works at the level of a WorkerPool, and a single Worker
    must be abstracted into a WorkerPool before it can be used by the
    Scheduler.

    However, to schedule a task across its `Worker`s, developers can provide
    a `Scheduler` instance that is in charge of scheduling the placed task.

    Args:
        name (`str`): A name assigned to this WorkerPool.
        id (`UUID`): The ID of the particular WorkerPool.
    """
    def __init__(self, name: str, workers: Optional[Sequence[Worker]],
                 scheduler: Optional[Type[BaseScheduler]]):
        self._name = name
        self._workers = {worker.id: worker for worker in workers}
        self._scheduler = scheduler
        self._id = uuid.uuid4()

    def add_workers(self, workers: Sequence[Worker]):
        """Adds the given set of `Worker`s to this `WorkerPool`.

        Args:
            workers (`Sequence[Worker]`): The workers to be added.
        """
        for worker in workers:
            if worker.id in self._workers:
                print("Skipping adding Worker with ID: {} because it already\
                        exists.".format(worker.id))
            else:
                self._workers[worker.id] = worker

    def place_task(self, task: Task):
        """Places the task on this `WorkerPool`.

        The caller must ensure that the `WorkerPool` has enough resources to
        execute this task before invoking this method. Further, the
        `WorkerPool` is in charge of executing the task across its workers.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.

        Raises:
            `ValueError` if the task could not be placed due to insufficient
            resources.
        """
        placement = None
        if self._scheduler is not None:
            # If a scheduler was provided, get a task placement from it.
            _, placement = self._scheduler.schedule(
                                [task],  # Only this task is available.
                                None,    # No task graph.
                                self._workers)
        else:
            # If there was no scheduler, find the first worker that can
            # accomodate the task given its resource requirements.
            for _id, _worker in self._workers:
                if _worker.can_accomodate_task(task):
                    placement = _id
                    break

        if placement is None:
            raise ValueError("The task ({}) could not be placed.".format(task))
        else:
            self._workers[placement].place_task(task)

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s that are currently placed on this `WorkerPool`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        placed_tasks = []
        for _, _worker in self._workers:
            placed_tasks.extend(_worker.get_placed_tasks())
        return placed_tasks

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @property
    def workers(self):
        return self._workers

    def __str__(self):
        return "WorkerPool(name={}, id={})".format(self.name, self.id)

    def __repr__(self):
        return str(self)
