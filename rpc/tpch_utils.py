# Code adopted from decima-sim to use their profiles of TPC-H queries

import ast
import json
import yaml
import os
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from data.tpch_loader import get_all_stage_info_for_query


TPCH_PARENT_DIR = "/home/dgarg39/erdos-scheduling-simulator/profiles/workload/tpch/"


def get_base_tpch_graph_structure(query_num):
    with open(os.path.join(TPCH_PARENT_DIR, "queries.yaml")) as f:
        tpch_query_yaml = yaml.load(f, Loader=yaml.FullLoader)

    # Extract the graph structure for the given query number
    query_graph = None
    for graph in tpch_query_yaml["graphs"]:
        if graph["name"] == f"Q{query_num}":
            query_graph = graph["graph"]
            break

    if query_graph is None:
        raise ValueError(f"Query number {query_num} not found in the YAML file")

    # Convert the graph structure to a format suitable for nx.DiGraph
    query_dependency = []
    for node in query_graph:
        if "children" in node:
            for child in node["children"]:
                query_dependency.append((node["name"], child))
        else:
            # Ensure each tuple has two elements by adding a dummy node
            query_dependency.append((node["name"], None))

    # Remove any tuples where the second element is None
    query_dependency = [edge for edge in query_dependency if edge[1] is not None]

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
