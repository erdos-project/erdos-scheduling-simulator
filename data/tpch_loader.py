import os
import math
import json
import sys
import random

from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

import absl
import numpy as np
import yaml

from utils import EventTime, setup_logging
from workload import (
    Workload,
    WorkProfile,
    TaskGraph,
    Job,
    JobGraph,
    ExecutionStrategy,
    ExecutionStrategies,
    Resource,
    Resources,
)

from .base_workload_loader import BaseWorkloadLoader


class TpchLoader(BaseWorkloadLoader):
    """Loads the TPCH trace from the provided file

    Args:
        path (`str`): Path to a YAML file specifying the TPC-H query DAGs
        flags (`absl.flags`): The flags used to initialize the app, if any
    """

    def __init__(self, path: str, flags: "absl.flags") -> None:
        self._flags = flags
        self._logger = setup_logging(
            name=self.__class__.__name__,
            log_dir=flags.log_dir,
            log_file=flags.log_file_name,
            log_level=flags.log_level,
        )
        self._rng_seed = flags.random_seed
        self._rng = random.Random(self._rng_seed)
        if flags.workload_update_interval > 0:
            self._workload_update_interval = flags.workload_update_interval
        else:
            self._workload_update_interval = EventTime(sys.maxsize, EventTime.Unit.US)

        # Set up task graph generators
        with open(path, "r") as f:
            workload_data = yaml.safe_load(f)
        task_graph_generators = {}
        for query in workload_data["graphs"]:
            query_name = query["name"]
            graph = query["graph"]
            gen = self.make_task_graph_generator(
                query_name=query_name,
                graph=graph,
            )
            task_graph_generators[query_name] = gen
        self._task_graph_generators = task_graph_generators

        # Gather release times
        release_policy = self._make_release_policy()
        release_times = release_policy.get_release_times(
            completion_time=EventTime(self._flags.loop_timeout, EventTime.Unit.US)
        )

        # Sample queries to be released
        query_nums = [
            self._rng.randint(1, len(self._task_graph_generators))
            for _ in range(self._flags.override_num_invocation)
        ]

        self._query_nums_and_release_times = list(zip(query_nums, release_times))
        self._current_release_pointer = 0

        # Initialize workload
        self._workload = Workload.empty(flags)

    def _make_release_policy(self):
        release_policy_args = {}
        if self._flags.override_release_policy == "periodic":
            release_policy_args = {
                "period": EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
            }
        elif self._flags.override_release_policy == "fixed":
            release_policy_args = {
                "period": EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
                "num_invocations": self._flags.override_num_invocation,
            }
        elif self._flags.override_release_policy == "poisson":
            release_policy_args = {
                "rate": self._flags.override_poisson_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
            }
        elif self._flags.override_release_policy == "gamma":
            release_policy_args = {
                "rate": self._flags.override_poisson_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
                "coefficient": self._flags.override_gamma_coefficient,
            }
        elif self._flags.override_release_policy == "fixed_gamma":
            release_policy_args = {
                "variable_arrival_rate": self._flags.override_poisson_arrival_rate,
                "base_arrival_rate": self._flags.override_base_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
                "coefficient": self._flags.override_gamma_coefficient,
            }
        else:
            raise NotImplementedError(
                f"Release policy {self._flags.override_release_policy} not implemented."
            )

        return make_release_policy(
            self._flags.override_release_policy,
            release_policy_args,
            self._rng,
            self._rng_seed,
            (
                self._flags.randomize_start_time_min,
                self._flags.randomize_start_time_max,
            ),
        )

    def make_task_graph_generator(
        self,
        query_name: str,
        graph: List[Dict[str, Any]],
    ) -> Callable[[int, EventTime, EventTime], TaskGraph]:
        def h(idx: int, current_time: EventTime, start_time: EventTime):
            # Construct a JobGraph
            job_graph = JobGraph(name=f"{query_name}[{idx}]")
            query_num = int(query_name[1:])
            profiler_data = get_all_stage_info_for_query(
                query_num,
                self._flags.tpch_profile_type,
                self._flags.tpch_dataset_size,
                self._flags.tpch_max_executors_per_job,
            )
            name_to_job = {}
            for node in graph:
                worker_profile = self.make_work_profile(
                    profiler_data=profiler_data,
                    query_name=query_name,
                    node_name=node["name"],
                )
                job = Job(
                    name=node["name"],
                    profile=worker_profile,
                )
                name_to_job[node["name"]] = job
                job_graph.add_job(job=job)
            for node in graph:
                job = name_to_job[node["name"]]
                if "children" in node:
                    for child in node["children"]:
                        if child not in name_to_job:
                            raise ValueError(
                                f"Child {child} of {node['name']} was "
                                f"not present in the graph."
                            )
                        child_job = name_to_job[child]
                        job_graph.add_child(job, child_job)

            # Construct TaskGraph from JobGraph
            task_graph = job_graph.get_next_task_graph(
                start_time=start_time,
                _flags=self._flags,
            )

            # Update deadline
            critical_path = task_graph.get_longest_path(
                weights=lambda task: (task.slowest_execution_strategy.runtime.time)
            )
            critical_path_time = (
                sum(
                    [t.slowest_execution_strategy.runtime for t in critical_path],
                    start=EventTime.zero(),
                )
                .to(EventTime.Unit.US)
                .time
            )
            deadline_variance_factor = (
                1.0
                + (
                    self._rng.randint(
                        self._flags.min_deadline_variance,
                        self._flags.max_deadline_variance,
                    )
                )
                / 100
            )
            task_graph_slo_time = math.ceil(
                critical_path_time * deadline_variance_factor
            )
            for task in task_graph.get_nodes():
                deadline = EventTime(
                    start_time.time + task_graph_slo_time, unit=EventTime.Unit.US
                )
                task.update_deadline(deadline)

            return task_graph

        return h

    def make_work_profile(
        self, profiler_data: Dict[int, Dict[str, Any]], query_name: str, node_name: str
    ) -> WorkProfile:
        profile = profiler_data[int(node_name)]

        profiled_task_slots = profile["num_tasks"]
        profiled_runtime = math.ceil(profile["avg_task_duration_ms"] / 1e3)

        if profiled_task_slots > self._flags.tpch_max_executors_per_job:
            num_slots = self._flags.tpch_max_executors_per_job
            runtime = math.ceil(
                (profiled_task_slots * profiled_runtime)
                / self._flags.tpch_max_executors_per_job
            )
            self._logger.debug(
                "%s@%s: num_slots (%s) > tpch_max_executors_per_job (%s). Converted "
                "(slots,runtime) from (%s,%s) to (%s, %s)",
                node_name,
                query_name,
                profiled_task_slots,
                self._flags.tpch_max_executors_per_job,
                profiled_task_slots,
                profiled_runtime,
                num_slots,
                runtime,
            )
        else:
            num_slots = profiled_task_slots
            runtime = profiled_runtime

        if runtime < self._flags.tpch_min_task_runtime:
            _runtime = runtime
            runtime = max(self._flags.tpch_min_task_runtime, _runtime)
            self._logger.debug(
                "%s@%s: runtime (%s) < tpch_min_task_runtime (%s). Converted "
                "(slots,runtime) from (%s,%s) to (%s, %s)",
                node_name,
                query_name,
                _runtime,
                self._flags.tpch_min_task_runtime,
                num_slots,
                _runtime,
                num_slots,
                runtime,
            )

        resources = Resources(
            resource_vector={
                Resource(name="Slot", _id="any"): num_slots,
            },
        )
        execution_strategies = ExecutionStrategies()
        execution_strategies.add_strategy(
            strategy=ExecutionStrategy(
                resources=resources,
                batch_size=1,
                runtime=EventTime(runtime, EventTime.Unit.US),
            ),
        )
        return WorkProfile(
            name=f"{query_name}_{node_name}_execution_profile",
            execution_strategies=execution_strategies,
        )

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        # Reset rng if this is the first workload. This is to ensure we have
        # parity with how jobs are spawned in Spark
        if self._current_release_pointer == 0:
            self._rng = random.Random(self._rng_seed)

        to_release = []
        while (
            self._current_release_pointer < len(self._query_nums_and_release_times)
            and self._query_nums_and_release_times[self._current_release_pointer][1]
            <= current_time + self._workload_update_interval
        ):
            to_release.append(
                self._query_nums_and_release_times[self._current_release_pointer]
            )
            self._current_release_pointer += 1

        if (
            self._current_release_pointer >= len(self._query_nums_and_release_times)
            and len(to_release) == 0
        ):
            # Nothing left to release
            return None

        for i, (q, t) in enumerate(to_release):
            query_name = f"Q{q}"
            task_graph = self._task_graph_generators[query_name](
                idx=i,
                current_time=current_time,
                start_time=t,
            )
            self._workload.add_task_graph(task_graph)

        return self._workload


def make_release_policy(
    release_policy, release_policy_args, rng, seed, randomize_start_time=(0, 0)
):
    # Check that none of the arg values are None
    assert all([val is not None for val in release_policy_args.values()])

    # Construct the release policy
    start_time = EventTime(
        time=rng.randint(*randomize_start_time),
        unit=EventTime.Unit.US,
    )
    release_policy = getattr(JobGraph.ReleasePolicy, release_policy)(
        start=start_time, rng_seed=seed, **release_policy_args
    )

    return release_policy


# TODO: make configurable
TPCH_SUBDIR = "100g/"
DECIMA_TPCH_DIR = (
    "/home/dgarg39/erdos-scheduling-simulator/profiles/workload/tpch/decima/"
)
CLOUDLAB_TPCH_DIR = (
    "/home/dgarg39/erdos-scheduling-simulator/profiles/workload/tpch/cloudlab/"
)


class SetWithCount(object):
    """
    allow duplication in set
    """

    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]


def pre_process_task_duration(task_duration):
    # remove fresh durations from first wave
    clean_first_wave = {}
    for e in task_duration["first_wave"]:
        clean_first_wave[e] = []
        fresh_durations = SetWithCount()
        # O(1) access
        for d in task_duration["fresh_durations"][e]:
            fresh_durations.add(d)
        for d in task_duration["first_wave"][e]:
            if d not in fresh_durations:
                clean_first_wave[e].append(d)
            else:
                # prevent duplicated fresh duration blocking first wave
                fresh_durations.remove(d)


def get_all_stage_info_for_query(query_num, profile_type, dataset_size, max_executors):
    stage_info = {}
    if profile_type == "Decima":
        stage_info = use_decima_tpch_profile(query_num, dataset_size)
    elif profile_type == "Cloudlab":
        stage_info = use_cloudlab_profile(query_num, dataset_size, max_executors)
    else:
        raise ValueError(f"Invalid profile type: {profile_type}")

    return stage_info


def use_cloudlab_profile(query_num, dataset_size, max_executors):
    cloudlab_profile_json = os.path.join(
        CLOUDLAB_TPCH_DIR, "cloudlab_22query_tpch_profiles.json"
    )
    with open(cloudlab_profile_json, "r") as file:
        data = json.load(file)

    query_key_to_extract = (
        "tpch_q"
        + str(query_num)
        + "_"
        + str(dataset_size)
        + "g"
        + "_maxCores_"
        + str(max_executors)
    )
    required_query_profile = data[query_key_to_extract]

    stage_info = {}

    for i, stage_profile in enumerate(required_query_profile):
        curr_stage = {
            "stage_id": i,
            "num_tasks": stage_profile["num_tasks"],
            "avg_task_duration_ms": round(stage_profile["average_runtime_ms"]),
        }
        stage_info[i] = curr_stage

    return stage_info


def use_decima_tpch_profile(query_num, dataset_size):
    task_durations = np.load(
        os.path.join(
            DECIMA_TPCH_DIR, dataset_size, "task_duration_" + str(query_num) + ".npy"
        ),
        allow_pickle=True,
    ).item()

    num_nodes = len(task_durations)

    stage_info = {}

    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration["first_wave"]))
        # NOTE: somehow only picks the first element {2: [n_tasks_in_ms]}

        num_tasks = len(task_duration["first_wave"][e]) + len(
            task_duration["rest_wave"][e]
        )

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for t in task_duration["first_wave"].values() for i in t]
            + [i for t in task_duration["rest_wave"].values() for i in t]
            + [i for t in task_duration["fresh_durations"].values() for i in t]
        )

        # NOTE: Runtime per task is given in milliseconds
        curr_stage = {
            "stage_id": n,
            "num_tasks": num_tasks,
            "avg_task_duration_ms": round(rough_duration),
        }
        stage_info[n] = curr_stage

    return stage_info
