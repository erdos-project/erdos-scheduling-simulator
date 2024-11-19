import os
import math
import json
import sys
import random

from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path

import absl
import numpy as np
import yaml
import networkx as nx

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


class TpchLoader:
    """Construct TPC-H task graph from a query profile

    Args:
        path (`str`): Path to a YAML file specifying the TPC-H query DAGs
        flags (`absl.flags`): The flags used to initialize the app, if any

    """

    def __init__(
        self,
        path: Path,
        flags: "absl.flags",
    ):
        self._logger = setup_logging(
            name=self.__class__.__name__,
            log_dir=flags.log_dir,
            log_file=flags.log_file_name,
            log_level=flags.log_level,
        )
        self._flags = flags

        # Load the TPC-H DAG structures
        with open(path, "r") as f:
            workload_data = yaml.safe_load(f)
        self._graphs = {}
        for query in workload_data["graphs"]:
            query_num = int(query["name"][1:])
            self._graphs[query_num] = query["graph"]

    def make_task_graph(
        self,
        id: str,
        query_num: int,
        release_time: EventTime,
        dependencies: Optional[List[Dict[str, Any]]] = None,
        profile_type: Optional[str] = None,
        dataset_size: Optional[int] = None,
        max_executors_per_job: Optional[int] = None,
        min_task_runtime: Optional[int] = None,
        runtime_unit: EventTime.Unit = EventTime.Unit.US,
    ) -> Tuple[TaskGraph, Dict[int, int]]:
        if profile_type is None:
            profile_type = self._flags.tpch_profile_type
        if dataset_size is None:
            dataset_size = self._flags.tpch_dataset_size
        if max_executors_per_job is None:
            max_executors_per_job = self._flags.tpch_max_executors_per_job
        if min_task_runtime is None:
            min_task_runtime = self._flags.tpch_min_task_runtime

        # Normalize dependencies
        if dependencies is None:
            dependencies = self._graphs[query_num]
            deps_mapping = None
        else:
            deps_mapping = self.__map_dependencies(query_num, dependencies)
            for node in dependencies:
                node["name"] = deps_mapping[node["name"]]
                if "children" in node:
                    node["children"] = [deps_mapping[c] for c in node["children"]]
            self._logger.info(
                f"Mapped dependencies for TPC-H query {query_name(query_num)} as {deps_mapping}."
            )

        # Construct a JobGraph
        job_graph = JobGraph(name=task_graph_name(query_num, id))
        profiler_data = get_all_stage_info_for_query(
            query_num,
            profile_type,
            dataset_size,
            max_executors_per_job,
        )
        name_to_job = {}
        for node in dependencies:
            worker_profile = self.__make_work_profile(
                profiler_data=profiler_data,
                query_num=query_num,
                node_name=node["name"],
                max_executors_per_job=max_executors_per_job,
                min_task_runtime=min_task_runtime,
                runtime_unit=runtime_unit,
            )
            job = Job(
                name=node["name"],
                profile=worker_profile,
            )
            name_to_job[node["name"]] = job
            job_graph.add_job(job=job)
        for node in dependencies:
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
            start_time=release_time,
            _flags=self._flags,
        )

        self._logger.info(
            f"Constructed TaskGraph for TPC-H query {query_name(query_num)}."
        )

        return task_graph, deps_mapping

    def __make_work_profile(
        self,
        profiler_data: Dict[int, Dict[str, Any]],
        query_num: int,
        node_name: str,
        max_executors_per_job: int,
        min_task_runtime: int,
        runtime_unit: EventTime,
    ) -> WorkProfile:
        profile = profiler_data[int(node_name)]

        profiled_task_slots = profile["num_tasks"]
        profiled_runtime = math.ceil(profile["avg_task_duration_ms"] / 1e3)

        if profiled_task_slots > max_executors_per_job:
            num_slots = max_executors_per_job
            runtime = math.ceil(
                (profiled_task_slots * profiled_runtime) / max_executors_per_job
            )
            self._logger.debug(
                "%s@%s: num_slots (%s) > max_executors_per_job (%s). Converted "
                "(slots,runtime) from (%s,%s) to (%s, %s)",
                node_name,
                query_name(query_num),
                profiled_task_slots,
                max_executors_per_job,
                profiled_task_slots,
                profiled_runtime,
                num_slots,
                runtime,
            )
        else:
            num_slots = profiled_task_slots
            runtime = profiled_runtime

        if runtime < min_task_runtime:
            _runtime = runtime
            runtime = max(min_task_runtime, _runtime)
            self._logger.debug(
                "%s@%s: runtime (%s) < min_task_runtime (%s). Converted "
                "(slots,runtime) from (%s,%s) to (%s, %s)",
                node_name,
                query_name(query_num),
                _runtime,
                min_task_runtime,
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
                runtime=EventTime(runtime, runtime_unit),
            ),
        )
        return WorkProfile(
            name=f"{query_name(query_num)}_{node_name}_execution_profile",
            execution_strategies=execution_strategies,
        )

    def __map_dependencies(self, query_num: int, deps: List[Dict[str, Any]]):
        def deps_to_nx_graph(deps: List[Dict[str, Any]]):
            query_dependency = []
            for node in deps:
                if "children" in node:
                    for child in node["children"]:
                        query_dependency.append((node["name"], child))
                else:
                    # Ensure each tuple has two elements by adding a dummy node
                    query_dependency.append((node["name"], None))

            # Remove any tuples where the second element is None
            query_dependency = [
                edge for edge in query_dependency if edge[1] is not None
            ]

            # convert job structure into a nx graph
            nx_deps = nx.DiGraph(query_dependency)

            return nx_deps

        def are_structurally_same(graph1, graph2):
            # Step 1: Check if both graphs have the same number of vertices
            if len(graph1.nodes) != len(graph2.nodes):
                return False, None

            # Step 2: Check if there exists a bijection between the vertices
            #         of the two graphs such that their adjacency relationships match
            for mapping in nx.isomorphism.GraphMatcher(
                graph1, graph2
            ).isomorphisms_iter():
                # Check if the adjacency relationships match
                if all(v in mapping for u, v in graph1.edges):
                    # graph structures match
                    # mapping is a dict {key=original-stage-id, val=app-stage-id}
                    # we reverse reversed mapping from app-stage-id to orig-stage-id
                    reversed_mapping = {v: k for k, v in mapping.items()}
                    return True, reversed_mapping

            return False, None

        base_deps = self._graphs[query_num]
        is_same, mapping = are_structurally_same(
            deps_to_nx_graph(base_deps), deps_to_nx_graph(deps)
        )

        if not is_same:
            raise ValueError(
                f"Structure of dependencies provided for query number {query_num} does not match that of canonical dependencies. Provided: {deps}. Canonical: {base_deps}"
            )

        return mapping

    @property
    def num_queries(self) -> int:
        return len(self._graphs)


class TpchWorkloadLoader(BaseWorkloadLoader):
    """Construct a TPC-H query workload

    Args:
        flags (`absl.flags`): The flags used to initialize the app, if any
    """

    def __init__(self, flags: "absl.flags") -> None:
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

        # Instantiate tpch loader
        self._tpch_loader = TpchLoader(path=flags.tpch_query_dag_spec, flags=flags)

        # Gather release times
        release_policy = self.__make_release_policy()
        release_times = release_policy.get_release_times(
            completion_time=EventTime(self._flags.loop_timeout, EventTime.Unit.US)
        )

        # Sample queries to be released
        query_nums = [
            self._rng.randint(1, self._tpch_loader.num_queries)
            for _ in range(self._flags.override_num_invocation)
        ]

        self._query_nums_and_release_times = list(zip(query_nums, release_times))
        self._current_release_pointer = 0

        # Initialize workload
        self._workload = Workload.empty(flags)

    def __make_release_policy(self):
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
            task_graph, _ = self._tpch_loader.make_task_graph(
                id=str(i),
                query_num=q,
                release_time=t,
            )
            self._workload.add_task_graph(task_graph)

        return self._workload


def query_name(query_num: int) -> str:
    return f"Q{query_num}"


def task_graph_name(query_num: int, id: any) -> str:
    return f"{query_name(query_num)}[{id}]"


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
