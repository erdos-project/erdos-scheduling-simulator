# Code adopted from decima-sim to use their profiles of TPC-H queries

import ast
import json
import os
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from data import TpchLoader

HOME_TPCH_DIR = "../profiles/workload/tpch_decima/"
TPCH_SUBDIR = "2g/"


def get_all_stage_info_for_query(query_num):
    return TpchLoader.get_profiler_data_for_query(
        profile_path=os.path.join(HOME_TPCH_DIR, TPCH_SUBDIR),
        query_num=query_num,
    )


def get_base_tpch_graph_structure(query_num):
    # use query_num to read string from file
    with open(os.path.join(HOME_TPCH_DIR, "query_dag.json")) as f:
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
