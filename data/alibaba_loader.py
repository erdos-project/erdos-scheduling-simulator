import math
import pathlib
import pickle
import random
from collections import defaultdict
from typing import List, Mapping, Optional

import absl  # noqa: F401

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


class AlibabaLoader(object):
    """Loads the Alibaba trace from the provided file.

    Args:
        path (`str`): The path to a Pickle file containing the Alibaba trace.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self, path: str, _flags: Optional["absl.flags"] = None) -> None:
        self._path = path
        self._flags = _flags

        extension = pathlib.Path(path).suffix.lower()
        if extension != ".pkl":
            raise ValueError(f"Invalid extension {extension} for Alibaba trace.")

        with open(self._path, "rb") as pickled_file:
            self.workload = AlibabaLoader.process_pickled_data(
                pickle.load(pickled_file), self._flags
            )

    @staticmethod
    def process_pickled_data(
        data: Mapping[str, List[str]], flags: "absl.flags"
    ) -> Workload:
        """Processes the data loaded from the Pickle file.

        Args:
            data (`Mapping[str, List[Task]]`): The data loaded from the Pickle file.

        Returns:
            A processed version of the data.
        """
        job_graphs = {}
        for job_graph_name, job_tasks in data.items():
            # Create the ReleasePolicy.
            release_policy = None
            start_time = EventTime(
                time=random.randint(
                    flags.randomize_start_time_min, flags.randomize_start_time_max
                ),
                unit=EventTime.Unit.US,
            )
            if flags.override_release_policy == "periodic":
                if flags.override_arrival_period == 0:
                    raise ValueError(
                        "Arrival period must be specified for periodic release policy."
                    )
                release_policy = JobGraph.ReleasePolicy.periodic(
                    period=EventTime(flags.override_arrival_period, EventTime.Unit.US),
                    start=start_time,
                )
            elif flags.override_release_policy == "fixed":
                if flags.override_arrival_period == 0:
                    raise ValueError(
                        "Arrival period must be specified for fixed release policy."
                    )
                release_policy = JobGraph.ReleasePolicy.fixed(
                    period=EventTime(flags.override_arrival_period, EventTime.Unit.US),
                    num_invocations=flags.override_num_invocations,
                    start=start_time,
                )
            elif flags.override_release_policy == "poisson":
                release_policy = JobGraph.ReleasePolicy.poisson(
                    rate=flags.override_poisson_arrival_rate,
                    num_invocations=flags.override_num_invocations,
                    start=start_time,
                )
            elif flags.override_release_policy == "gamma":
                release_policy = JobGraph.ReleasePolicy.gamma(
                    rate=flags.override_poisson_arrival_rate,
                    num_invocations=flags.override_num_invocations,
                    coefficient=flags.override_gamma_coefficient,
                    start=start_time,
                )
            else:
                raise NotImplementedError(
                    f"Release policy {flags.override_release_policy} not implemented."
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

            job_graphs[job_graph_name] = JobGraph(
                name=job_graph_name,
                jobs=jobs_to_children,
                release_policy=release_policy,
                deadline_variance=(
                    flags.min_deadline_variance,
                    flags.max_deadline_variance,
                ),
            )
        workload = Workload.from_job_graphs(job_graphs=job_graphs, _flags=flags)
        return workload
