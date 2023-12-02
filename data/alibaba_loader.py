import json
import math
import os
import pathlib
import pickle
import random
import sys
from collections import defaultdict
from typing import List, Mapping, Optional, Sequence

import absl

from utils import EventTime, setup_csv_logging, setup_logging
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

from .base_workload_loader import BaseWorkloadLoader


class AlibabaLoader(BaseWorkloadLoader):
    """Loads the Alibaba trace from the provided file.

    Args:
        path (`str`): The path to a Pickle file containing the Alibaba trace,
            or a folder containing multiple Pickle files.
        workload_interval (`EventTime`): The interval at which to release new
            Workloads.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        path: str,
        workload_interval: EventTime,
        flags: "absl.flags",
    ):
        self._path = path
        self._flags = flags
        self._job_data_generator = self._initialize_job_data_generator()
        self._job_graphs: Mapping[str, JobGraph] = {}
        self._rng_seed = flags.random_seed
        self._rng = random.Random(self._rng_seed)
        self._release_times = self._construct_release_times()
        self._current_release_pointer = 0
        self._workload_update_interval = (
            workload_interval
            if not workload_interval.is_invalid()
            else EventTime(sys.maxsize, EventTime.Unit.US)
        )
        self._workload = Workload.empty(flags)

        if self._flags:
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__,
                log_dir=self._flags.log_dir,
                log_file=self._flags.csv_file_name,
            )

            self._task_cpu_divisor = int(self._flags.alibaba_loader_task_cpu_divisor)
            self._task_duration_multipler = self._flags.alibaba_task_duration_multiplier
        else:
            self._csv_logger = setup_csv_logging(
                name=self.__class__.__name__, log_file=None
            )
            self._log_dir = os.getcwd()
            self._task_cpu_divisor = 25
            self._task_duration_multipler = 1

    def _construct_release_times(self):
        """Construct the release times of the jobs in the workload.

        Returns:
            A list of release times of the jobs in the workload.
        """
        # Create the ReleasePolicy.
        release_policy = None
        start_time = EventTime(
            time=self._rng.randint(
                self._flags.randomize_start_time_min,
                self._flags.randomize_start_time_max,
            ),
            unit=EventTime.Unit.US,
        )
        if self._flags.override_release_policy == "periodic":
            if self._flags.override_arrival_period == 0:
                raise ValueError(
                    "Arrival period must be specified for periodic release policy."
                )
            release_policy = JobGraph.ReleasePolicy.periodic(
                period=EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif self._flags.override_release_policy == "fixed":
            if self._flags.override_arrival_period == 0:
                raise ValueError(
                    "Arrival period must be specified for fixed release policy."
                )
            release_policy = JobGraph.ReleasePolicy.fixed(
                period=EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
                num_invocations=self._flags.override_num_invocations,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif self._flags.override_release_policy == "poisson":
            release_policy = JobGraph.ReleasePolicy.poisson(
                rate=self._flags.override_poisson_arrival_rate,
                num_invocations=self._flags.override_num_invocations,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        elif self._flags.override_release_policy == "gamma":
            release_policy = JobGraph.ReleasePolicy.gamma(
                rate=self._flags.override_poisson_arrival_rate,
                num_invocations=self._flags.override_num_invocations,
                coefficient=self._flags.override_gamma_coefficient,
                start=start_time,
                rng_seed=self._rng_seed,
            )
        else:
            raise NotImplementedError(
                f"Release policy {self._flags.override_release_policy} not implemented."
            )
        return release_policy.get_release_times(
            completion_time=EventTime(self._flags.loop_timeout, EventTime.Unit.US)
        )

    def _initialize_job_data_generator(self):
        """
        Initialize the job generator from the Alibaba trace file.
        """
        if os.path.isdir(self._path):
            file_paths = [
                os.path.join(self._path, filename)
                for filename in os.listdir(self._path)
                if filename.endswith(".pkl")
            ]
        elif os.path.isfile(self._path):
            extension = pathlib.Path(self._path).suffix.lower()
            if extension != ".pkl":
                raise ValueError(f"Invalid extension {extension} for Alibaba trace.")
            file_paths = [self._path]
        else:
            raise FileNotFoundError(f"No such file or directory: {self._path}")

        def job_data_generator():
            for file_path in file_paths:
                with open(file_path, "rb") as pickled_file:
                    data: Mapping[str, List[str]] = pickle.load(pickled_file)
                    for job_graph_name, job_tasks in data.items():
                        try:
                            self._job_graphs[
                                job_graph_name
                            ] = self._convert_job_data_to_job_graph(
                                job_graph_name, job_tasks
                            )
                        except Exception as e:
                            print(f"Error converting {job_graph_name}: {e}")
                            pass
                yield

        return job_data_generator()

    def _convert_job_data_to_job_graph(
        self, job_graph_name: str, job_tasks: List[str]
    ) -> JobGraph:
        """
        Convert the raw job data to a Job object.

        This method should be implemented according to the specifics of the
        Alibaba trace file format and your Job class.
        """
        # Create the individual Job instances corresponding to each Task.
        task_name_to_simulator_job_mapping = {}
        for task in job_tasks:
            job_resources = Resources(
                resource_vector={
                    # Note: We divide the CPU by 25 instead of 100 because this
                    # would intorduce more variance into the resource/slots usage.
                    # We used to divide by 100, but the majority of the tasks
                    # would end up using 1 slot, which is not very interesting and 
                    # makes no chance for DAG_Sched to do effective packing that 
                    # would beat EDF by a significant margin.
                    Resource(name="Slot", _id="any"): int(math.ceil(task.cpu / self._task_cpu_divisor)),
                }
            )
            job_name = task.name.split("_")[0]
            job_runtime = EventTime(int(math.ceil(task.duration * self._task_duration_multipler)), EventTime.Unit.US)
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
            deadline_variance=(
                self._flags.min_deadline_variance,
                self._flags.max_deadline_variance,
            ),
        )

    def get_next_jobs(self, start_time_offset: int = 0) -> Sequence[JobGraph]:
        print(f"{start_time_offset=}")
        if self._batch_size <= 0:
            return [
                self._convert_job_data_to_job_graph(
                    job_graph_name, job_tasks, start_time_offset
                )
                for job_graph_name, job_tasks in self._job_data_generator
            ]
        else:
            batch: List[JobGraph] = []
            try:
                for _ in range(self._batch_size):
                    job_graph_name, job_tasks = next(self._job_data_generator)
                    job_graph = self._convert_job_data_to_job_graph(
                        job_graph_name, job_tasks, start_time_offset
                    )
                    batch.append(job_graph)
            except StopIteration:
                pass
            return batch

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        # Load the next batch of jobs into our mapping.
        try:
            next(self._job_data_generator)
        except StopIteration:
            pass

        # Get the release times that fit within the range of the current_time and the
        # current_time + workload_interval.
        released_taskgraph_times = []
        while (
            self._current_release_pointer < len(self._release_times)
            and self._release_times[self._current_release_pointer]
            <= current_time + self._workload_update_interval
        ):
            released_taskgraph_times.append(
                self._release_times[self._current_release_pointer]
            )
            self._current_release_pointer += 1

        if (
            self._current_release_pointer >= len(self._release_times)
            and len(released_taskgraph_times) == 0
        ):
            # We are at the end of the times, and we didn't release anything this time.
            return None
        else:
            # Choose a random JobGraph and convert it to a TaskGraph to be released.
            task_release_index = 0
            while task_release_index < len(released_taskgraph_times):
                job_graph = self._rng.choice(list(self._job_graphs.values()))
                task_graph = job_graph.get_next_task_graph(
                    start_time=released_taskgraph_times[task_release_index],
                    _flags=self._flags,
                )
                if task_graph is not None:
                    self._workload.add_task_graph(task_graph)
                    self._csv_logger.info(f"{current_time.time},TASK_GRAPH_RELEASE,{task_graph.release_time.time},{task_graph.deadline.time},{task_graph.name},{len(task_graph.get_nodes())}")
                    task_release_index += 1
            return self._workload
