import uuid
from workload.resources import Resources
from typing import Optional, Sequence


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

    Args:
        name (`str`): A name assigned to this WorkerPool.
        id (`UUID`): The ID of the particular WorkerPool.
    """
    def __init__(self, name: str, workers: Optional[Sequence[Worker]]):
        self._name = name
        self._workers = {}
        for worker in workers:
            self._workers[worker.id] = worker
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
