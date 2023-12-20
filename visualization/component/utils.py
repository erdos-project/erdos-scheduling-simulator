import heapq
import pickle
import sys
from collections import namedtuple
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from streamlit_agraph import Config, Edge, Node, agraph

sys.path.append("..")
from data.csv_reader import CSVReader
from utils import DisjointedIntervals


@dataclass
class CSV_Data:
    df_task_graphs: pd.DataFrame
    df_tasks: pd.DataFrame
    df_worker_pools: pd.DataFrame
    worker_pool_stats: dict[str, tuple[float, float]]

    completed_task_graph_run_time: int
    canceled_task_graph_run_time: int
    miss_deadline_task_graph_run_time: int


@st.cache_data
def get_csv_data(csv_file_path):
    csv_reader = CSVReader([csv_file_path])

    worker_pools = csv_reader.get_worker_pools(csv_file_path)
    df_worker_pools = pd.DataFrame(worker_pools)
    df_worker_pools["resources_str"] = [
        str(resources) for resources in df_worker_pools["resources"]
    ]

    worker_pool_stats = csv_reader.get_worker_pool_utilizations(csv_file_path)
    task_graphs = csv_reader.get_task_graph(csv_file_path).values()
    tasks = csv_reader.get_tasks(csv_file_path)
    (
        completed_task_graph_run_time,
        canceled_task_graph_run_time,
        miss_deadline_task_graph_run_time,
    ) = csv_reader.get_time_spent_on_completed_canceled_miss_deadline_task_graph(
        csv_file_path
    )

    df_task_graphs = pd.DataFrame(task_graphs)
    df_tasks = pd.DataFrame(tasks)

    # Add a new column "task_graph_miss_deadline" in df_tasks that joins
    # df_tasks["task_graph"] on df_task_graphs["name"] and use the value of
    # df_task_graphs["deadline_miss_detected_at"].notnull()
    df_task_graphs_renamed = df_task_graphs.rename(
        columns={
            "name": "task_graph_name",
            "deadline_miss_detected_at": "task_graph_deadline_miss_detected_at",
        }
    )
    df_tasks = df_tasks.merge(
        df_task_graphs_renamed[
            ["task_graph_name", "task_graph_deadline_miss_detected_at"]
        ],
        how="left",
        left_on="task_graph",
        right_on="task_graph_name",
    )
    df_tasks["task_graph_miss_deadline"] = df_tasks[
        "task_graph_deadline_miss_detected_at"
    ].notnull()
    df_tasks.drop(["task_graph_name"], axis=1, inplace=True)

    return CSV_Data(
        df_task_graphs=df_task_graphs,
        df_tasks=df_tasks.drop(["timestamp", "task_id"], axis=1)[
            [
                "task_graph",
                "name",
                "release_time",
                "deadline",
                "window_to_execute",
                "runtime",
                "slowest_execution_time",
                "placement_time",
                "completion_time",
                "slack",
                "start_time",
                "cancelled",
                "cancelled_at",
                "missed_deadline",
                "deadline_miss_detected_at",
                "task_graph_miss_deadline",
                "task_graph_deadline_miss_detected_at",
                "skipped_times",
                "placements",
            ]
        ],
        df_worker_pools=df_worker_pools,
        worker_pool_stats=worker_pool_stats,
        completed_task_graph_run_time=sum(completed_task_graph_run_time.values()),
        canceled_task_graph_run_time=sum(canceled_task_graph_run_time.values()),
        miss_deadline_task_graph_run_time=sum(
            miss_deadline_task_graph_run_time.values()
        ),
    )


Task = namedtuple(
    "Task",
    field_names="name,job,instances,status,start_time,end_time,duration,cpu,mem",
)


@st.cache_data
def get_original_trace_data(trace_file_path):
    with open(trace_file_path, "rb") as pickled_file:
        return pickle.load(pickled_file)


@st.cache_data(experimental_allow_widgets=True)
def visualize_task_graph(task_graph_id, df_tasks, trace_data):
    st.write(
        "Green means 'Finished.' Red means 'Cancelled.' \
        Orange means 'Finished but missed deadline.'"
    )
    if task_graph_id.split("@")[0] not in trace_data:
        st.write(f'Task Graph "{task_graph_id}" not found in trace data')
        return
    if len(df_tasks[df_tasks["task_graph"] == task_graph_id]) == 0:
        st.write(f'Task Graph "{task_graph_id}" not found in this run')
        return

    tasks = trace_data[task_graph_id.split("@")[0]]
    task_to_children = {}
    id_to_task_name = {}
    for task in tasks:
        task_name = task.name.split("_")[0]
        task_to_children[task_name] = []
        id_to_task_name[task_name[1:]] = task_name

    # Create the JobGraph.
    for task in tasks:
        task_and_parents = task.name.split("_", 1)
        if len(task_and_parents) == 1:
            # This job has no parent, add an empty list.
            task_to_children[task_and_parents[0]].extend([])
        else:
            # This job has children, find them from the list.
            task_name = task_and_parents[0]
            parent_tasks = set(task_and_parents[1].split("_"))
            for parent_task in parent_tasks:
                task_to_children[id_to_task_name[parent_task]].append(task_name)

    nodes, edges = [], []
    for task_name in id_to_task_name.values():
        row = df_tasks[
            (df_tasks["name"] == task_name) & (df_tasks["task_graph"] == task_graph_id)
        ]
        nodes.append(
            Node(
                id=task_name,
                # title get displayed when you hover over the node
                title=f"release={row['release_time'].item()}, \
                        place={row['placement_time'].item()}, \
                        actual_runtime={row['runtime'].item()}, \
                        slowest_runtime={row['slowest_execution_time'].item()}",
                label=f"{task_name},"
                + f"slowest_runtime={row['slowest_execution_time'].item()}",
                size=25,
                shape="dot",
                color="red"
                if row["cancelled"].item()
                else "orange"
                if row["missed_deadline"].item()
                else "green",
            )
        )

    for task_name, children in task_to_children.items():
        for child in children:
            edges.append(Edge(source=task_name, label="", target=child, type="arrow"))

    config = Config(
        width=1000,
        height=800,
        nodeSpacing=200,
        sortMethod="directed",
        shakeTowards="roots",
        directed=True,
        physics=False,
        hierarchical=False,
    )
    return agraph(nodes=nodes, edges=edges, config=config)


@st.cache_data
def plot_task_placement_timeline_chart(
    df_worker_pools: pd.DataFrame, df_tasks: pd.DataFrame
):
    st.write("### Task Placement Timeline")
    for i, worker_pool in df_worker_pools.iterrows():
        st.write(f"#### {worker_pool['name']}")
        # disjoint_intervals_pools is a list of DisjointedIntervals
        # Think of each DisjointedIntervals as a horizontal line where
        # non-overlapping tasks can be drawn. When there are tasks being
        # run concurrently, we need multiple horizontal lines to draw them.
        disjoint_intervals_pools = [DisjointedIntervals()]
        task_placements = []
        for i, task in df_tasks[df_tasks["start_time"].notnull()].iterrows():
            start_time = task["placement_time"]
            runtime = task["runtime"]
            interval = (start_time, start_time + runtime - 1)
            for placement in task["placements"]:
                for resource in placement["resources_used"]:
                    # Find the y_index to place this task on the timeline chart
                    y_index = 0
                    for j, disjoint_intervals in enumerate(disjoint_intervals_pools):
                        if not disjoint_intervals.overlap(interval):
                            disjoint_intervals.add(interval)
                            y_index = j
                            break
                    else:
                        disjoint_intervals_pools.append(DisjointedIntervals())
                        disjoint_intervals_pools[-1].add(interval)
                        y_index = len(disjoint_intervals_pools) - 1

                    task_placements.append(
                        {
                            "y_index": str(y_index),
                            "task": task["name"],
                            "task_graph": task["task_graph"],
                            "release": task["release_time"],
                            "deadline": task["deadline"],
                            "start": start_time,
                            "end": start_time + runtime,
                            "runtime": runtime,
                            "resource": f'{resource["name"]}, {resource["quantity"]}',
                            "cancelled": task["cancelled"],
                            "missed_deadline": task["missed_deadline"],
                            "task_graph_miss_deadline": task[
                                "task_graph_miss_deadline"
                            ],
                        }
                    )

        source = pd.DataFrame(task_placements)

        chart = (
            alt.Chart(source)
            .mark_bar()
            .encode(
                x="start",
                x2="end",
                y="y_index",
                detail=[
                    "task",
                    "task_graph",
                    "release",
                    "deadline",
                    "runtime",
                    "resource",
                    "cancelled",
                    "missed_deadline",
                    "task_graph_miss_deadline",
                ],
                color="task_graph",
            )
            .interactive()
        )

        st.altair_chart(chart, theme="streamlit", use_container_width=True)


@st.cache_data
def plot_task_placement_per_slot_timeline_chart(
    df_worker_pools: pd.DataFrame, df_tasks: pd.DataFrame
):
    st.write("### Task Slots Placement Timeline")
    for _, worker_pool in df_worker_pools.iterrows():
        st.write(f"#### {worker_pool['name']}")

        disjoint_intervals_per_resource = {}
        for resource in worker_pool["resources"]:
            disjoint_intervals_per_resource[resource["name"]] = [
                DisjointedIntervals() for _ in range(int(resource["quantity"]))
            ]

        task_placements = []
        df_tasks = df_tasks[df_tasks["placement_time"].notnull()]
        df_tasks = df_tasks.sort_values(by=["placement_time"])
        for _, task in df_tasks.iterrows():
            start_time = task["placement_time"]
            runtime = task["runtime"]
            interval = (start_time, start_time + runtime - 1)
            for placement in task["placements"]:
                for resource_used in placement["resources_used"]:
                    # Assume that quantity is discrete integer for slots
                    quantity = int(resource_used["quantity"])

                    # Find the disjoint_intervals such that the gap after
                    # inserted the new interval and its left interval
                    # are the smallest.

                    # A maxheap of (-gap, slot_id)
                    placement_candidates = find_placement_slot_candidates(
                        disjoint_intervals_per_resource,
                        interval,
                        resource_used,
                        quantity,
                    )

                    if len(placement_candidates) < quantity:
                        raise Exception(
                            f"Not enough slots for {task['name']} "
                            f"from {task['task_graph']}. {start_time=}, {runtime=}"
                        )

                    for _, slot_id in placement_candidates:
                        disjoint_intervals_per_resource[resource["name"]][slot_id].add(
                            interval
                        )
                        task_placements.append(
                            {
                                "slot_id": f"{resource_used['name']}-{slot_id}",
                                "task": task["name"],
                                "task_graph": task["task_graph"],
                                "release": task["release_time"],
                                "deadline": task["deadline"],
                                "start": task["start_time"],
                                "end": task["start_time"] + task["runtime"],
                                "runtime": task["runtime"],
                                "resource": f'{resource_used["name"]},'
                                + f'{resource_used["quantity"]}',
                                "cancelled": task["cancelled"],
                                "missed_deadline": task["missed_deadline"],
                                "task_graph_miss_deadline": task[
                                    "task_graph_miss_deadline"
                                ],
                            }
                        )

        source = pd.DataFrame(task_placements)

        chart = (
            alt.Chart(source)
            .mark_bar()
            .encode(
                x="start",
                x2="end",
                y="slot_id",
                detail=[
                    "task",
                    "task_graph",
                    "release",
                    "deadline",
                    "runtime",
                    "resource",
                    "cancelled",
                    "missed_deadline",
                    "task_graph_miss_deadline",
                ],
                color="task_graph",
            )
            .interactive()
        )

        st.altair_chart(chart, theme="streamlit", use_container_width=True)


def find_placement_slot_candidates(
    disjoint_intervals_per_resource, interval, resource_used, quantity
):
    """
    This function finds the placement slot candidates for a task. We
    want the placement to be as packed as possible.
    See https://leetcode.com/problems/k-closest-points-to-origin/
    """
    placement_candidates = []
    for slot_id, disjoint_intervals in enumerate(
        disjoint_intervals_per_resource[resource_used["name"]]
    ):
        if not disjoint_intervals.overlap(interval):
            if len(placement_candidates) < quantity:
                heapq.heappush(
                    placement_candidates,
                    (
                        -disjoint_intervals.placement_gap_with_left_interval(interval),
                        slot_id,
                    ),
                )
            elif (
                disjoint_intervals.placement_gap_with_left_interval(interval)
                < -placement_candidates[0][0]
            ):
                heapq.heappushpop(
                    placement_candidates,
                    (
                        -disjoint_intervals.placement_gap_with_left_interval(interval),
                        slot_id,
                    ),
                )

    return placement_candidates


@st.cache_data
def plot_resource_utilization_timeline_chart(
    worker_pool_stats: dict[str, tuple[float, float]],
    smoothing_window_size=10,
):
    def smooth_data(y, window_size):
        """Applies a moving average filter to smooth the data."""
        window = np.ones(window_size) / window_size
        return np.convolve(y, window, mode="same")

    resource_color = {
        "GPU": "red",
        "CPU": "green",
        "All_Slots": "red",
        "Slot_1": "blue",
        "Slot_2": "orange",
    }
    fig, ax = plt.subplots(figsize=(10, 2))

    resource_types: set[str] = set()
    for wp_stats in worker_pool_stats:
        for resource in wp_stats.resource_utilizations.keys():
            resource_types.add(resource)
    num_resource_type = len(resource_types)

    # Calculate the heights of utilization for each resource, after
    # normalization.
    resource_used_heights = {
        resource: [
            stat.resource_utilizations[resource][0]
            / sum(stat.resource_utilizations[resource])
            for stat in worker_pool_stats
        ]
        for resource in resource_types
    }

    sim_times_sec = [stat.simulator_time / 1000000 for stat in worker_pool_stats]

    # Combine slot_1 and slot_2 into a total "slot"
    if "All_Slots" not in resource_types:
        resource_types.add("All_Slots")
        resource_used_heights["All_Slots"] = []
        for resource_type in resource_types:
            if resource_type.startswith("Slot"):
                for i, h in enumerate(resource_used_heights[resource_type]):
                    if i >= len(resource_used_heights["All_Slots"]):
                        resource_used_heights["All_Slots"].append(h)
                    else:
                        resource_used_heights["All_Slots"][i] += h

    # Plotting for this scheduler
    for resource_type in resource_types:
        smoothed_utilization = smooth_data(
            resource_used_heights[resource_type], smoothing_window_size
        )
        ax.plot(
            sim_times_sec,
            smoothed_utilization,
            color=resource_color[resource_type],
            label=f"{resource_type}" if i == 0 else "",
        )

    # Formatting the subplot
    ax.set_ylabel("Utilization", fontsize=16)
    ax.set_ylim(0, num_resource_type + 0.01)
    ax.set_title("Resource Utilization")
    st.pyplot(fig)
