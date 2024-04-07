# Code adopted from decima-sim to use their profiles of TPC-H queries

import ast
import json
import os
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

DECIMA_TPCH_DIR = "../profiles/workload/tpch/decima/"
TPCH_SUBDIR = "100g/"
CLOUDLAB_TPCH_DIR = "../profiles/workload/tpch/cloudlab/"


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
        print("## FATAL ERROR ## Could not find TPCH profile to use")

    return stage_info


def use_cloudlab_profile(query_num, dataset_size, max_executors):
    cloudlab_profile_json = os.path.join(
        CLOUDLAB_TPCH_DIR, "cloudlab_22query_tpch_profiles.json")
    with open(cloudlab_profile_json, 'r') as file:
        data = json.load(file)
    
    query_key_to_extract = ("tpch_q" + str(query_num) +
                            "_" + str(dataset_size) +
                            "_maxCores_" + str(max_executors))
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


def get_base_tpch_graph_structure(query_num):
    # use query_num to read string from file
    with open(os.path.join(DECIMA_TPCH_DIR, "query_dag.json")) as f:
        tpch_query_json = json.load(f)

    # get query dependency from file
    query_dependency = ast.literal_eval(tpch_query_json["query_number"][str(query_num)])

    # convert job structure into a nx graph
    base_tpch_graph = nx.DiGraph(query_dependency)

    # return the job nx graph for query
    return base_tpch_graph


def get_graph_from_deps(dependencies):
    # parse dependencies to get it in list of tuples
    # construct the TaskGraph from the dependencies.
    task_graph_structure: Mapping[int, Sequence[int]] = {}
    for task_dependency in dependencies:
        task_graph_structure[task_dependency.key.id] = [
            task_id for task_id in task_dependency.children_ids
        ]

    # convert our TaskGraph into a nx friendly notation
    all_edges_in_app = []
    for parent in task_graph_structure.keys():
        for child in task_graph_structure[parent]:
            all_edges_in_app.append((parent, child))

    # construct nx graph
    given_app_tpch_graph = nx.DiGraph(all_edges_in_app)

    # return the graph
    return given_app_tpch_graph


def are_structurally_same(graph1, graph2):
    # Step 1: Check if both graphs have the same number of vertices
    if len(graph1.nodes) != len(graph2.nodes):
        print(
            f"DAG structure mismatch! Graph1 has "
            f"{graph1.nodes} while Graph2 has {graph2.nodes}"
        )
        return False, None

    # Step 2: Check if there exists a bijection between the vertices
    #         of the two graphs such that their adjacency relationships match
    for mapping in nx.isomorphism.GraphMatcher(graph1, graph2).isomorphisms_iter():
        # Check if the adjacency relationships match
        if all(v in mapping for u, v in graph1.edges):
            # graph structures match
            # mapping is a dict {key=original-stage-id, val=app-stage-id}
            # we reverse reversed mapping from app-stage-id to orig-stage-id
            reversed_mapping = {v: k for k, v in mapping.items()}

            return True, reversed_mapping

    print("DAG structure mismatch! No mapping could be found")
    return False, None


def verify_and_relable_tpch_app_graph(query_num, dependencies):
    # get nx graphs from base tpch file and application dependencies
    base_tpch_graph = get_base_tpch_graph_structure(query_num)
    app_graph = get_graph_from_deps(dependencies)

    # verify that the graphs are isomorphic
    # returns true if same_structure, along with stage_id_mapping wrt base tpch file
    same_structure, stage_id_mapping = are_structurally_same(base_tpch_graph, app_graph)

    # return with stage_id_mapping back to assign correct runtime and num_executors
    return same_structure, stage_id_mapping