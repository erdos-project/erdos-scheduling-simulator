import uuid
from collections import defaultdict
from typing import Mapping, Sequence, Optional


class Job(object):
    """A `Job` represents a particular Operator in an ERDOS AV pipeline.

    A `Job` is a static entity and along with other `Job`s creates a `JobGraph`
    which represents the static structure of the computation.

    The `Job`s release a set of `Task`s that are runtime entities and must be
    scheduled by a `Scheduler`.

    Args:
        name: The name of the ERDOS operator that corresponds to this job.
    """
    def __init__(self, name: str):
        self._name = name
        self._id = uuid.uuid1()

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    def __str__(self):
        return "Job(name={}, id={})".format(self.name, self.id)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)


class JobGraph(object):
    """A `JobGraph` represents a directed Operator graph that makes up an
    ERDOS AV pipeline.

    A `JobGraph` is a static entity that defines the relationship between the
    different `Job`s of the application.
    """
    def __init__(self, jobs: Optional[Mapping[Job, Sequence[Job]]] = None):
        self._job_graph = defaultdict(list) if jobs is None else jobs

    def add_job(self, job: Job, children: Optional[Sequence[Job]]):
        """Adds the job to the graph along with the given children.

        Args:
            job (`Job`): The job to be added to the graph.
            children (`Sequence[Job]`): The children of this job, if any.
        """
        if job in self._job_graph:
            print("The job {job} is already in the graph. Skipping.".
                  format(job=job))
        else:
            _children = [] if children is None else children
            self._job_graph[job].extend(_children)

    def add_child(self, job: Job, child: Job):
        """Adds a child to a `Job` in the job graph.

        Args:
            job (`Job`): The job, to which the child needs to be added.
            child (`Job`): The child job to be added.
        """
        self._job_graph[job].append(child)

    def get_children(self, job: Job) -> Sequence[Job]:
        """Retrieves the children jobs of the given job.

        Args:
            job (`Job`): The job whose children need to be retrieved.

        Returns:
            A Sequence[Job] containing the children of the given job.
        """
        if job not in self._job_graph:
            raise ValueError("No job with the ID: {} exists in the graph".
                             format(job.id))
        return self._job_graph[job]

    def __str__(self):
        return "JobGraph({})".format(self._job_graph)

    def __repr__(self):
        return str(self)
