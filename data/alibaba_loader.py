import math
import os
import pathlib
import pickle
import random
from collections import defaultdict
from typing import Generator, List, Mapping, Optional

import absl
from data.workload_loader_v2 import WorkloadLoaderV2  # noqa: F401

from utils import EventTime
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    JobGraph,
    Resource,
    Resources,
    Workload,
    WorkProfile,
)


class AlibabaLoader(WorkloadLoaderV2):
    """Loads the Alibaba trace from the provided file.

    Args:
        path (`str`): The path to a Pickle file containing the Alibaba trace, 
            or a folder containing multiple Pickle files.
        batch_size (`int`): The batch size to use when loading jobs. If 0, 
            all jobs will be loaded at once.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self, path: str, batch_size: int = 0, _flags: Optional["absl.flags"] = None):
        self._batch_size = batch_size
        self._path = path
        self._flags = _flags
        self._job_generator = None

    def _initialize_job_generator(self):
        """
        Initialize the job generator from the Alibaba trace file.
        """
        if os.path.isdir(self._path):
            file_paths = [os.path.join(self._path, filename) for filename in os.listdir(self._path) if filename.endswith('.pkl')]
        elif os.path.isfile(self._path):
            extension = pathlib.Path(self._path).suffix.lower()
            if extension != ".pkl":
                raise ValueError(f"Invalid extension {extension} for Alibaba trace.")
            file_paths = [self._path]
        else:
            raise FileNotFoundError(f"No such file or directory: {self._path}")

        def job_generator():
            for file_path in file_paths:
                with open(file_path, "rb") as pickled_file:
                    data: Mapping[str, List[str]] = pickle.load(pickled_file)
                    for job_graph_name, job_tasks in data.items():
                        yield self._convert_job_data_to_job(job_graph_name, job_tasks)

        self._job_generator = job_generator()

    def _convert_job_data_to_job(self, job_graph_name: str, job_tasks: List[str]) -> JobGraph:
        """
        Convert the raw job data to a Job object.

        This method should be implemented according to the specifics of the
        Alibaba trace file format and your Job class.
        """
        # Create the ReleasePolicy.
        release_policy = None
        start_time = EventTime(
            time=random.randint(
                self._flags.randomize_start_time_min, self._flags.randomize_start_time_max
            ),
            unit=EventTime.Unit.US,
        )
        if self._flags.override_release_policy == "periodic":
            if self._flags.override_arrival_period == 0:
                raise ValueError(
                    "Arrival period must be specified for periodic release policy."
                )
            release_policy = JobGraph.ReleasePolicy.periodic(
                period=EventTime(self._flags.override_arrival_period, EventTime.Unit.US),
                start=start_time,
            )
        elif self._flags.override_release_policy == "fixed":
            if self._flags.override_arrival_period == 0:
                raise ValueError(
                    "Arrival period must be specified for fixed release policy."
                )
            release_policy = JobGraph.ReleasePolicy.fixed(
                period=EventTime(self._flags.override_arrival_period, EventTime.Unit.US),
                num_invocations=self._flags.override_num_invocations,
                start=start_time,
            )
        elif self._flags.override_release_policy == "poisson":
            release_policy = JobGraph.ReleasePolicy.poisson(
                rate=self._flags.override_poisson_arrival_rate,
                num_invocations=self._flags.override_num_invocations,
                start=start_time,
            )
        elif self._flags.override_release_policy == "gamma":
            release_policy = JobGraph.ReleasePolicy.gamma(
                rate=self._flags.override_poisson_arrival_rate,
                num_invocations=self._flags.override_num_invocations,
                coefficient=self._flags.override_gamma_coefficient,
                start=start_time,
            )
        else:
            raise NotImplementedError(
                f"Release policy {self._flags.override_release_policy} not implemented."
            )
        # Create the individual Job instances corresponding to each Task.
        task_name_to_simulator_job_mapping = {}
        for task in job_tasks:
            job_resources = Resources(
                resource_vector={
                    Resource(name="Slot", _id="any"): int(
                        math.ceil(task.cpu / 100)
                    ),
                }
            )
            job_name = task.name.split("_")[0]
            job_runtime = EventTime(
                int(math.ceil(task.duration)), EventTime.Unit.US
            )
            task_name_to_simulator_job_mapping[job_name] = Job(
                name=job_name,
                profile=WorkProfile(
                    name="SlotPolicyFor{}".format(job_name),
                    execution_strategies=ExecutionStrategies(
                        [
                            ExecutionStrategy(
                                resources=job_resources,
                                batch_size=1,
                                runtime=job_runtime,
                            )
                        ]
                    ),
                ),
            )

        # Create the JobGraph.
        jobs_to_children = defaultdict(list)
        for task in job_tasks:
            job_and_parents = task.name.split("_", 1)
            if len(job_and_parents) == 1:
                # This job has no parent, add an empty list.
                jobs_to_children[
                    task_name_to_simulator_job_mapping[job_and_parents[0]]
                ].extend([])
            else:
                # This job has children, find them from the list.
                current_job = job_and_parents[0]
                parents = set(job_and_parents[1].split("_"))
                for (
                    parent_job_name,
                    parent_job,
                ) in task_name_to_simulator_job_mapping.items():
                    if parent_job_name[1:] in parents:
                        jobs_to_children[parent_job].append(
                            task_name_to_simulator_job_mapping[current_job]
                        )

        return JobGraph(
            name=job_graph_name,
            jobs=jobs_to_children,
            release_policy=release_policy,
            deadline_variance=(
                self._flags.min_deadline_variance,
                self._flags.max_deadline_variance,
            ),
        )

    def workloads(self) -> Generator[Workload, None, None]:
        if self._job_generator is None:
            self._initialize_job_generator()
        
        if self._batch_size <= 0:
            # Yield all jobs at once
            job_graphs = {job.name: job for job in self._job_generator}
            yield Workload.from_job_graphs(job_graphs=job_graphs, _flags=self._flags)
            return # The generator is exhausted
        else:
            batch: List[JobGraph] = []
            try:
                for _ in range(self._batch_size):
                    job = next(self._job_generator)
                    batch.append(job)
                yield Workload.from_job_graphs(job_graphs={job.name: job for job in batch}, _flags=self._flags)
            except StopIteration:
                if batch:
                    # Yield any remaining jobs in the last batch
                    yield Workload.from_job_graphs(job_graphs={job.name: job for job in batch}, _flags=self._flags)
                return  # The generator is exhausted