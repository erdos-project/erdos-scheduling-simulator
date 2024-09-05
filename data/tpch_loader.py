import sys

from typing import Any, Dict, List, Optional
from pathlib import Path

import absl #noqa: F401
import numpy as np
import yaml

from utils import EventTime
from workload import (
    Workload,
    WorkProfile,
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
        _flags (`absl.flags`): The flags used to initialize the app, if any
    """
    def __init__(self, path: str, _flags: Optional["absl.flags"] = None) -> None:
        if _flags:
            self._loop_timeout = _flags.loop_timeout
            self._workload_profile_path = _flags.workload_profile_path
        else:
            self._loop_timeout = EventTime(time=sys.maxsize, unit=EventTime.Unit.US)
            self._workload_profile_path = "./profiles/workload/tpch/decima/2g"

        with open(path, "r") as f:
            workload_data = yaml.safe_load(f)

        job_graphs = {}
        for query in workload_data["graphs"]:
            query_name = query["name"]
            graph = query["graph"]
            job_graph = TpchLoader.make_job_graph(
                query_name=query_name,
                graph=graph,
                profile_path=self._workload_profile_path,
            )
            job_graphs[query_name] = job_graph

        workload = Workload.from_job_graphs(job_graphs)
        workload.populate_task_graphs(completion_time=self._loop_timeout)
        self._workloads = iter([workload])


    @staticmethod
    def make_job_graph(query_name: str, graph: List[Dict[str, Any]], profile_path: str) -> JobGraph:
        job_graph = JobGraph(
            name=query_name,

            # TODO: make configurable
            release_policy=JobGraph.ReleasePolicy.fixed(
                period=EventTime(30, EventTime.Unit.US),
                num_invocations=10,
                start=EventTime(0, EventTime.Unit.US),
            ),

            # TODO: make configurable
            deadline_variance=(0,0),
        )

        query_num = int(query_name[1:])
        profiler_data = TpchLoader.get_profiler_data_for_query(profile_path, query_num)

        name_to_job = {}
        for node in graph:
            worker_profile = TpchLoader.load_query_profile(
                profiler_data=profiler_data,
                query_name=query_name,
                node_name=node["name"],
            )
            job = Job(name=node["name"], profile=worker_profile)
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

        return job_graph


    @staticmethod
    def load_query_profile(profiler_data: Dict[int, Dict[str, Any]], query_name: str, node_name: str) -> WorkProfile:
        profile = profiler_data[int(node_name)]
        resources = Resources(
            resource_vector={
                Resource(name="Slot", _id="any"): profile["num_tasks"],
            },
        )
        execution_strategies = ExecutionStrategies()
        execution_strategies.add_strategy(
            strategy=ExecutionStrategy(
                resources=resources,
                batch_size=1,
                runtime=EventTime(profile["avg_task_duration"], EventTime.Unit.US),
            ),
        )
        return WorkProfile(
            name=f"{query_name}_{node_name}_execution_profile",
            execution_strategies=execution_strategies,
        )


    @staticmethod
    def get_profiler_data_for_query(profile_path: str, query_num: int) -> Dict[int, Dict[str, Any]]:
        def pre_process_task_duration(task_duration):
            # remove fresh durations from first wave
            clean_first_wave = {}
            for e in task_duration["first_wave"]:
                clean_first_wave[e] = []
                fresh_durations = SetWithCount()
                for d in task_duration["fresh_durations"][e]:
                    fresh_durations.add(d)
                for d in task_duration["first_wave"][e]:
                    if d not in fresh_durations:
                        clean_first_wave[e].append(d)
                    else:
                        # prevent duplicated fresh duration blocking first wave
                        fresh_durations.remove(d)

        task_durations = np.load(
            Path(profile_path) / f"task_duration_{query_num}.npy",
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

            curr_stage = {
                "stage_id": n,
                "num_tasks": num_tasks,
                "avg_task_duration": round(rough_duration),
            }
            stage_info[n] = curr_stage

        return stage_info


    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        try:
            return next(self._workloads)
        except StopIteration:
            return None


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

