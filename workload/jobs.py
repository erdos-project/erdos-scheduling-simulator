import random
import uuid
from collections import defaultdict, deque
from typing import Mapping, Optional, Sequence


class Job(object):
    """A `Job` represents a particular Operator in an ERDOS AV pipeline.

    A `Job` is a static entity and along with other `Job`s creates a `JobGraph`
    which represents the static structure of the computation.

    The `Job`s release a set of `Task`s that are runtime entities and must be
    scheduled by a `Scheduler`.

    Args:
        name: The name of the ERDOS operator that corresponds to this job.
        pipelined (`bool`): True if job's tasks from different timestamps can run
            in parallel.
    """

    def __init__(self, name: str, pipelined: bool = False):
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._pipelined = pipelined

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def pipelined(self):
        return self._pipelined

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return f"Job(name={self.name}, id={self.id}, pipelined={self.pipelined})"

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

    def __init__(self, jobs: Optional[Mapping[Job, Sequence[Job]]] = {}):
        self._job_graph = defaultdict(list)
        self.__parent_job_graph = defaultdict(list)
        for job, children in jobs.items():
            self._job_graph[job].extend(children)
            for child in children:
                self._job_graph[child].extend([])
                self.__parent_job_graph[child].append(job)

    def add_job(self, job: Job, children: Optional[Sequence[Job]] = []):
        """Adds the job to the graph along with the given children.

        Args:
            job (`Job`): The job to be added to the graph.
            children (`Sequence[Job]`): The children of this job, if any.
        """
        self._job_graph[job].extend(children)

        # Add the children into the graph too so the length is correct.
        for child in children:
            self._job_graph[child].extend([])
            self.__parent_job_graph[child].append(job)

    def add_child(self, job: Job, child: Job):
        """Adds a child to a `Job` in the job graph.

        Args:
            job (`Job`): The job, to which the child needs to be added.
            child (`Job`): The child job to be added.

        Raises:
            ValueError if the `job` is not in the graph already.
        """
        if job not in self._job_graph:
            raise ValueError(f"{job} not in job graph.")
        self._job_graph[job].append(child)
        self._job_graph[child].extend([])
        self.__parent_job_graph[child].append(job)

    def get_children(self, job: Job) -> Sequence[Job]:
        """Retrieves the children jobs of the given job.

        Args:
            job (`Job`): The job whose children need to be retrieved.

        Returns:
            A Sequence[Job] containing the children of the given job.
        """
        if job not in self._job_graph:
            raise ValueError(f"No job with the ID: {job.id} exists in the graph")
        return self._job_graph[job]

    def get_parents(self, job: Job) -> Sequence[Job]:
        """Retrieves the parents of the given Job.

        Args:
            job (`Job`): The job to retrieve the parents of.

        Returns:
            The parents of the given job.
        """
        if job not in self._job_graph:
            return []
        else:
            return self.__parent_job_graph[job]

    def pipeline_source_operators(self):
        """Ensures that the source operators pipeline tasks."""
        for job in self.get_source_jobs():
            job._pipelined = True

    def get_source_jobs(self) -> Sequence[Job]:
        """Retrieves the source jobs from the given graph.

        Returns:
            The source jobs from the given graph.
        """
        source_jobs = []
        for job in self._job_graph.keys():
            if len(self.__parent_job_graph[job]) == 0:
                source_jobs.append(job)
        return source_jobs

    def get_job_depth(self, job: Job) -> int:
        """Retrieves the depth of the given Job in the JobGraph.

        Args:
            job (`Job`): The job to check for a source Job.

        Returns:
            The depth of the Job with the depth of a source job being 1.
        """
        if job not in self._job_graph:
            raise ValueError(f"The job: {job} was not found in the graph.")
        if self.is_source_job(job):
            return 1

        return max(self.get_job_depth(parent) for parent in self.get_parents(job)) + 1

    def is_source_job(self, job: Job) -> bool:
        """Checks whether the given job is a source job.

        Args:
            job (`Job`): The job to check for a source Job.

        Returns:
            `True` if `job` is a source job, and `False` otherwise.
        """
        return len(self.get_parents(job)) == 0

    def __iter__(self):
        """Iterates through the JobGraph in a BFS manner."""
        visited_nodes = set()
        frontier = deque(self.get_source_jobs())
        while True:
            if len(frontier) == 0:
                break
            else:
                job = frontier.popleft()
                visited_nodes.add(job)
                for child in self.get_children(job):
                    if all(
                        parent in visited_nodes for parent in self.get_parents(child)
                    ):
                        frontier.append(child)
                yield job

    def __len__(self):
        return len(self._job_graph)

    def __str__(self):
        return f"JobGraph({self._job_graph})"

    def __repr__(self):
        return str(self)
