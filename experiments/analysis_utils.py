# flake8: noqa
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def calculate_arrival_rate_and_cv2(release_time: list[int]):
    release_time.sort()
    inter_arrival_times = np.diff(release_time)
    avg_inter_arrival_time = np.mean(inter_arrival_times)
    std_inter_arrival_time = np.std(inter_arrival_times)
    cv2 = (std_inter_arrival_time / avg_inter_arrival_time) ** 2
    return 1 / avg_inter_arrival_time, cv2


def find_all_file_paths(path, ends_with=".csv"):
    csv_file_paths = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(ends_with):
                csv_file_paths.append(os.path.join(path, filename))
            else:
                csv_file_paths += find_all_file_paths(
                    os.path.join(path, filename), ends_with
                )
    return csv_file_paths


def extract_variables_from_filename(filename):
    # Split the filename by underscores
    parts = filename.split("_")

    # Extract the variables based on your format
    replay_trace = parts[0]
    scheduler = parts[2]
    release_policy = parts[5]
    deadline_var = int(parts[9])
    dag_aware = parts[12] == "1"

    try:
        variable_arrival_rate = float(parts[16])
        cv2 = int(parts[19].split(".")[0])  # Assuming the file extension is .csv
    except:
        # Before 11/28 afternoon, I used a different format for the filename and
        # didn't include the arrival rate and CV2
        variable_arrival_rate = 10
        cv2 = 2

    if scheduler == "TetriSched":
        scheduler_time_discretization = int(parts[-1].split(".")[0])
        scheduler = f"TetriSched_time_dis_{scheduler_time_discretization}" + (
            "_DAG_aware" if dag_aware else ""
        )
    else:
        scheduler_time_discretization = None

    # Create a dictionary to store the extracted variables
    variables = {
        "trace": replay_trace,
        "release_policy": release_policy,
        "max_deadline_variance": deadline_var,
        "scheduler": scheduler,
        "DAG_aware": dag_aware,
        "scheduler_time_discretization": scheduler_time_discretization,
        "variable_arrival_rate": variable_arrival_rate,
        "cv2": cv2,
    }

    return variables


def extract_variables_from_filename_v2(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    variables = {}
    for line in lines:
        if "input_flag" in line:
            _, flag_name, flag_value = line.strip().split(",")
            if flag_name == "replay_trace":
                variables["trace"] = flag_value
            elif flag_name == "override_release_policy":
                variables["release_policy"] = flag_value
            elif flag_name == "min_deadline_variance":
                variables["min_deadline_variance"] = int(flag_value)
            elif flag_name == "max_deadline_variance":
                variables["max_deadline_variance"] = int(flag_value)
            elif flag_name == "scheduler":
                variables["scheduler"] = flag_value
            elif flag_name == "release_taskgraphs":
                variables["DAG_aware"] = flag_value.lower() == "true"
            elif flag_name == "scheduler_time_discretization":
                variables["scheduler_time_discretization"] = int(flag_value)
            elif flag_name == "override_poisson_arrival_rate":
                variables["variable_arrival_rate"] = float(flag_value)
            elif flag_name == "override_base_arrival_rate":
                variables["base_arrival_rate"] = float(flag_value)
            elif flag_name == "override_gamma_coefficient":
                variables["cv2"] = float(flag_value)
            elif flag_name == "workload_profile_path":
                variables["workload_profile_path"] = flag_value
            elif flag_name == "worker_profile_path":
                variables["worker_profile_path"] = flag_value
            elif flag_name == "scheduler_dynamic_discretization":
                variables["dynamic_discretization"] = flag_value.lower() == "true"
            elif flag_name == "scheduler_adaptive_discretization":
                variables["adaptive_discretization"] = flag_value.lower() == "true"
            elif flag_name == "scheduler_max_time_discretization":
                variables["max_time_discretization"] = int(flag_value)
            elif flag_name == "scheduler_max_occupancy_threshold":
                variables["max_occupancy_threshold"] = float(flag_value)
        else:
            break

    if variables["scheduler"] == "TetriSched":
        variables[
            "scheduler"
        ] = f'TetriSched_time_dis_{variables["scheduler_time_discretization"]}' + (
            "_DAG_aware" if variables["DAG_aware"] else ""
        )
    return variables


def extract_experiments_result(base_dir: str) -> pd.DataFrame:
    rows = []
    # Loop through each folder and process the CSV file
    for csv_file_path in find_all_file_paths(base_dir):
        file_name = csv_file_path.split(os.sep)[-1]
        if file_name == "libtetrisched_performance.csv":
            continue
        try:
            # Open the CSV file and read the last line
            with open(csv_file_path, "r") as file:
                lines = file.readlines()
                last_line = lines[-1]

            (
                end_time,
                _,
                finished_tasks,
                cancelled_tasks,
                missed_task_deadlines,
                finished_task_graphs,
                cancelled_task_graphs,
                missed_task_graph_deadlines,
            ) = last_line.split(",")

            row = extract_variables_from_filename_v2(csv_file_path)
            # row = extract_variables_from_filename(file_name)

            # Analyze SLO attainment and goodput
            slo_attainment = (
                int(finished_task_graphs) - int(missed_task_graph_deadlines)
            ) / (int(cancelled_task_graphs) + int(finished_task_graphs))
            row["slo_attainment"] = slo_attainment
            row["goodput"] = int(finished_tasks)
            row["csv_file_path"] = csv_file_path

            # Calculate the arrival rate and cv2
            release_times = []
            for line in lines:
                # if "TASK_RELEASE" in line:
                #     # Compute arrival rate and cv2 for "task" release time
                #     # event_time should be the actual release time
                #     event_time, _, task_name, _, task_intended_release_time, task_release_time, task_deadline, task_id, task_graph = line.strip().split(",")

                if "TASK_GRAPH_RELEASE" in line:
                    (
                        event_time,
                        _,
                        release_time,
                        deadline,
                        task_graph_name,
                        number_of_nodes,
                    ) = line.strip().split(",")
                    release_times.append(int(release_time))

            actual_arrival_rate, actual_cv2 = calculate_arrival_rate_and_cv2(
                release_times
            )
            row["actual_arrival_rate"] = actual_arrival_rate
            row["actual_cv2"] = actual_cv2
            row["num_invocation"] = len(release_times)

            rows.append(row)
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
        except Exception as e:
            print(f"An error occurred while processing {csv_file_path}: {str(e)}")
            # I want to remove the parent folder of the CSV file
            # print(f"Removing {os.path.dirname(csv_file_path)}")
            # shutil.rmtree(os.path.dirname(csv_file_path))

    return pd.DataFrame(rows)


def plot_slo_attainments(data: pd.DataFrame, extra_title: str = "", figsize=(20, 15)):
    # Define your unique values for the grid
    cv2_values = sorted(data["cv2"].unique())
    variable_arrival_rate_values = sorted(data["variable_arrival_rate"].unique())
    scheduler_values = [
        "TetriSched_time_dis_20",
        "TetriSched_time_dis_20_DAG_aware",
        "TetriSched_time_dis_10",
        "TetriSched_time_dis_10_DAG_aware",
        "TetriSched_time_dis_1",
        "TetriSched_time_dis_1_DAG_aware",
        "EDF",
    ]
    num_invocation = data["num_invocation"].unique()[0]
    # Number of schedulers
    n_schedulers = len(scheduler_values)

    # Create a subplot grid
    fig, axes = plt.subplots(
        len(variable_arrival_rate_values),
        len(cv2_values),
        figsize=figsize,
        sharey=True,
    )

    # Define the width of each bar and the spacing between them
    bar_width = 0.20
    spacing = 0.05
    group_width_factor = 2  # Increase this factor to widen the distance between groups

    # Collect handles and labels for the legend
    handles, labels = [], []

    # Iterate over each subplot and plot the data
    for i, variable_arrival_rate in enumerate(variable_arrival_rate_values):
        for j, cv2 in enumerate(cv2_values):
            if len(variable_arrival_rate_values) == 1 and len(cv2_values) == 1:
                ax = axes
            elif len(variable_arrival_rate_values) == 1:
                ax = axes[j]
            else:
                ax = axes[i][j]

            subset = data[
                (data["variable_arrival_rate"] == variable_arrival_rate)
                & (data["cv2"] == cv2)
            ]

            # Get unique deadline variances
            deadline_vars = sorted(subset["max_deadline_variance"].unique())
            x = np.arange(len(deadline_vars)) * group_width_factor  # Adjust x positions

            for k, scheduler in enumerate(scheduler_values):
                scheduler_data = subset[subset["scheduler"] == scheduler]
                # Calculate the position of each bar
                bar_positions = (
                    x - (n_schedulers * bar_width / 2) + (k * bar_width) + (spacing * k)
                )
                # Some bars may not exist for some schedulers
                slo_attainments = []
                for deadline_var in deadline_vars:
                    if (
                        len(
                            scheduler_data[
                                scheduler_data["max_deadline_variance"] == deadline_var
                            ]["slo_attainment"]
                        )
                        == 0
                    ):
                        slo_attainments.append(0)
                    else:
                        slo_attainments.append(
                            scheduler_data[
                                scheduler_data["max_deadline_variance"] == deadline_var
                            ]["slo_attainment"].item()
                        )

                ax.bar(bar_positions, slo_attainments, width=bar_width, label=scheduler)

            for c in ax.containers:
                labels = [f"{(v.get_height() * 100):.1f}" for v in c]
                ax.bar_label(c, labels=labels, label_type="edge", rotation=45, size=8)

            ax.set_xticks(x)
            ax.set_xticklabels(deadline_vars)

            # This is "task graph" arrival rate and cv2
            ax.set_title(
                f"Input Arrival Rate: {variable_arrival_rate}, CV2: {cv2} | Actual Arrival Rate: {subset['actual_arrival_rate'].mean():.3f}, CV2: {subset['actual_cv2'].mean():.3f}"
            )
            # ax.set_title(f"Actual Arrival Rate: {subset['actual_arrival_rate'].mean():.3f}, CV2: {subset['actual_cv2'].mean():.3f}")

            ax.set_xlabel("Max Deadline Variance")
            ax.set_ylabel("SLO Attainment")
            ax.set_ylim(top=1.01, bottom=0.0)

    # Adjust layout and add a super title
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.88
    )  # Adjust the bottom parameter to make space for the legend

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(labels),
    )

    plt.suptitle(
        f"SLO Attainment Comparison (min_deadline_var=10, num_invocation={num_invocation}) {extra_title}",
        size=16,
    )

    # Show the plot
    plt.show()


sys.path.append("..")
from data.csv_reader import CSVReader


def smooth_data(y, window_size):
    """Applies a moving average filter to smooth the data."""
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode="same")


# Obsolete
def analyze_resource_utilization_by_arrival_rate_and_cv2_and_max_deadline_var(
    csv_reader,
    df,
    variable_arrival_rate,
    cv2,
    max_deadline_variance,
    figure_size=(20, 20),
    axes_fontsize=16,
    smoothing_window_size=10,  # Size of the moving average window
    extra_title: str = "",
):
    num_invocation = df["num_invocation"].unique()[0]
    # Filter the DataFrame
    filtered_df = df[
        (df["variable_arrival_rate"] == variable_arrival_rate)
        & (df["cv2"] == cv2)
        & (df["max_deadline_variance"] == max_deadline_variance)
    ]
    num_schedulers = filtered_df["scheduler"].nunique()

    # Create subplots
    fig, axes = plt.subplots(num_schedulers, 1, figsize=figure_size, sharex=True)
    resource_color = {
        "GPU": "red",
        "CPU": "green",
        "All_Slots": "red",
        "Slot_1": "blue",
        "Slot_2": "orange",
    }

    # Iterate over each scheduler
    for i, (index, row) in enumerate(filtered_df.iterrows()):
        ax = axes[i] if num_schedulers > 1 else axes

        # Worker Pool statistics
        try:
            worker_pool_stats = csv_reader.get_worker_pool_utilizations(
                row["csv_file_path"]
            )
        except:
            print(
                f"Error while csv_reader.get_worker_pool_utilizations{row['csv_file_path']}"
            )
            continue

        # Find all the resource types in the system.
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
        ax.set_ylabel("Utilization", fontsize=axes_fontsize)
        ax.set_ylim(0, num_resource_type + 0.01)
        ax.set_title(f"{row['scheduler']}")

    # Common X-axis label
    plt.xlabel("Timeline [s]", fontsize=axes_fontsize)

    # Creating a common legend
    legend_elements = [
        Patch(facecolor=resource_color[rt], label=rt) for rt in resource_types
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize="small")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.suptitle(
        f"Resource Utilization for {variable_arrival_rate=}, {cv2=},  max deadline variance={max_deadline_variance}. Num invocations={num_invocation} {extra_title}"
    )

    # Display the plot
    plt.show()


# Obsolete
def analyze_resource_utilization_by_release_policy_and_max_deadline_var(
    csv_reader,
    df,
    release_policy,
    max_deadline_variance,
    figure_size=(20, 20),
    axes_fontsize=16,
    smoothing_window_size=10,  # Size of the moving average window
):
    # Filter the DataFrame
    filtered_df = df[
        (df["release_policy"] == release_policy)
        & (df["max_deadline_variance"] == max_deadline_variance)
    ]
    num_schedulers = filtered_df["scheduler"].nunique()

    # Create subplots
    fig, axes = plt.subplots(num_schedulers, 1, figsize=figure_size, sharex=True)
    # Have "Slot": "blue" for backward compatibility
    resource_color = {
        "GPU": "red",
        "CPU": "green",
        "Slot": "red",
        "Slot_1": "blue",
        "Slot_2": "orange",
    }

    # Iterate over each scheduler
    for i, (index, row) in enumerate(filtered_df.iterrows()):
        ax = axes[i] if num_schedulers > 1 else axes

        # Worker Pool statistics
        worker_pool_stats = csv_reader.get_worker_pool_utilizations(
            row["csv_file_path"]
        )

        # Find all the resource types in the system.
        resource_types: set[str] = set(["Slot"])
        for wp_stats in worker_pool_stats:
            for resource in wp_stats.resource_utilizations.keys():
                resource_types.add(resource)

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
        ax.set_ylabel("Utilization", fontsize=axes_fontsize)
        ax.set_ylim(0, 1.01)
        ax.set_title(f"{row['scheduler']}")

    # Common X-axis label
    plt.xlabel("Timeline [s]", fontsize=axes_fontsize)

    # Creating a common legend
    legend_elements = [
        Patch(facecolor=resource_color[rt], label=rt) for rt in resource_types
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize="small")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.suptitle(
        f"Resource Utilization for {release_policy} policy with max deadline variance={max_deadline_variance}. Num invocations=400"
    )

    # Display the plot
    plt.show()


def analyze_resource_utilization_by_arrival_rate_and_cv2(
    csv_reader,
    df,
    max_deadline_var,
    figure_size=(20, 20),
    axes_fontsize=16,
    smoothing_window_size=10,  # Size of the moving average window
    extra_title: str = "",
):
    # Unique values for variable_arrival_rate and cv2
    arrival_rates = sorted(df["variable_arrival_rate"].unique().tolist())
    cv2_values = sorted(df["cv2"].unique().tolist())

    # Create a superplot grid
    fig, axes = plt.subplots(
        len(arrival_rates),
        len(cv2_values),
        figsize=figure_size,
        sharex=True,
        sharey=True,
    )
    for r, variable_arrival_rate in enumerate(arrival_rates):
        for c, cv2 in enumerate(cv2_values):
            if len(arrival_rates) == 1 and len(cv2_values) == 1:
                ax = axes
            elif len(arrival_rates) == 1:
                ax = axes[c]
            else:
                ax = axes[r, c]

            # Filter the DataFrame for current variable_arrival_rate and cv2
            filtered_df = df[
                (df["variable_arrival_rate"] == variable_arrival_rate)
                & (df["cv2"] == cv2)
                & (df["max_deadline_variance"] == max_deadline_var)
            ]

            # Iterate over each scheduler in the filtered DataFrame
            for index, row in filtered_df.iterrows():
                # Worker Pool statistics
                try:
                    worker_pool_stats = csv_reader.get_worker_pool_utilizations(
                        row["csv_file_path"]
                    )
                except Exception as e:
                    print(
                        f"Error while csv_reader.get_worker_pool_utilizations{row['csv_file_path']}: {e}"
                    )
                    continue

                # Find all the resource types in the system.
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

                sim_times_sec = [
                    stat.simulator_time / 1000000 for stat in worker_pool_stats
                ]

                # Combine all slots into a total "slot"
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
                # for resource_type in resource_types:
                smoothed_utilization = smooth_data(
                    resource_used_heights["All_Slots"], smoothing_window_size
                )
                ax.plot(
                    sim_times_sec,
                    smoothed_utilization,
                    # color=resource_color[resource_type],
                    label=row["scheduler"],
                )

                # # Formatting the subplot
                ax.set_ylim(0, num_resource_type + 0.01)
                ax.legend()
                # ax.set_title(f"{row['scheduler']}")

                # Set subplot title or labels if necessary
                ax.set_ylabel("Utilization", fontsize=axes_fontsize)
                ax.set_xlabel("Time", fontsize=axes_fontsize)
                ax.set_title(
                    f'Arrival Rate: {variable_arrival_rate} v.s. {row["actual_arrival_rate"]:.3f}, CV2: {cv2} v.s. {row["actual_cv2"]:.3f}'
                )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    num_invocation = df["num_invocation"].unique()[0]
    plt.suptitle(
        f"Resource Utilization Analysis. {max_deadline_var=} {num_invocation=} {extra_title}"
    )

    # Display the plot
    plt.show()


def plot_resource_utilization(
    base_dir: str, extra_title: str = "", figure_size=(20, 20), packed: bool = True
):
    """
    If you have just one row, figure_size=(15, 8) is a good size.
    """
    # This function wraps analyze_resource_utilization_by_arrival_rate_and_cv2
    df = extract_experiments_result(base_dir)
    df = df.sort_values(
        by=[
            "variable_arrival_rate",
            "cv2",
            "scheduler_time_discretization",
            "scheduler",
            "DAG_aware",
            "max_deadline_variance",
        ]
    )
    # Filter out csv files that hasn't been completed yet
    csv_reader = CSVReader(df["csv_file_path"].tolist())

    if packed:
        for max_deadline_variance in df["max_deadline_variance"].unique():
            analyze_resource_utilization_by_arrival_rate_and_cv2(
                csv_reader,
                df,
                max_deadline_variance,
                figure_size=figure_size,
                extra_title=extra_title,
            )
    else:
        for variable_arrival_rate in df["variable_arrival_rate"].unique():
            for cv2 in df["cv2"].unique():
                for max_deadline_variance in df["max_deadline_variance"].unique():
                    analyze_resource_utilization_by_arrival_rate_and_cv2_and_max_deadline_var(
                        csv_reader,
                        df,
                        variable_arrival_rate=variable_arrival_rate,
                        cv2=cv2,
                        max_deadline_variance=max_deadline_variance,
                        figure_size=figure_size,
                        extra_title=extra_title,
                    )


def plot_solver_time(
    log_file_path_1: str,
    title_1: str,
    log_file_path_2: str = None,
    title_2: str = None,
    super_title: str = "Solver Time Microseconds",
):
    # Regular expression pattern to find 'solverTimeMicroseconds=' followed by a number
    pattern = r"solverTimeMicroseconds=(\d+)"

    solver_times_1, solver_times_2 = [], []

    # Process the first log file
    with open(log_file_path_1, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "solverTimeMicroseconds" in line:
                match = re.search(pattern, line)
                if match:
                    solver_time_microseconds = int(match.group(1))
                    solver_times_1.append(solver_time_microseconds)

    # Process the second log file, if provided
    if log_file_path_2 is not None:
        with open(log_file_path_2, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "solverTimeMicroseconds" in line:
                    match = re.search(pattern, line)
                    if match:
                        solver_time_microseconds = int(match.group(1))
                        solver_times_2.append(solver_time_microseconds)

    # Plotting
    plt.figure(figsize=(12, 6))

    if log_file_path_2 is None:
        # If only one file, plot a single histogram
        plt.hist(solver_times_1, bins=30, alpha=0.7, label=log_file_path_1)
        plt.legend([title_1])
    else:
        # Plot histograms for comparison
        plt.hist(solver_times_1, bins=30, alpha=0.7, label=log_file_path_1)
        plt.hist(solver_times_2, bins=30, alpha=0.7, label=log_file_path_2)
        plt.legend([title_1, title_2])

    plt.xlabel("Solver Time Microseconds")
    plt.ylabel("Frequency")
    plt.title(super_title)

    plt.show()

    # Optional: Plot boxplots for comparison
    plt.figure(figsize=(6, 6))
    data_to_plot = [solver_times_1]
    labels = [title_1]
    if log_file_path_2 is not None:
        data_to_plot.append(solver_times_2)
        labels.append(title_2)
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel("Solver Time Microseconds")
    plt.title(super_title)
    plt.show()


def extract_solver_time(base_dir: str):
    log_file_paths = find_all_file_paths(base_dir, ".log")
    solver_times = []
    for log_file_path in log_file_paths:
        if "TetriSched" not in os.path.basename(log_file_path):
            continue
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r"solverTimeMicroseconds=(\d+)", line)
                if match:
                    solver_time_microseconds = match.group(1)
                    solver_times.append(int(solver_time_microseconds))

    for p in range(0, 105, 5):
        print(f"{p}% percentile: {(np.percentile(solver_times, p) / 1e6):.2f} seconds")
