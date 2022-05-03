import os
import re
import sys
from collections import defaultdict
from operator import attrgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.patches import Patch
from tabulate import tabulate

from data import CSVReader
from utils import log_statistics, setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_list("csv_files", None, "List of CSV files containing experiment logs")
flags.mark_flag_as_required("csv_files")
flags.DEFINE_list("csv_labels", None, "List of labels to use for the experiment logs")
flags.mark_flag_as_required("csv_labels")
flags.DEFINE_string("output_dir", ".", "The directory to output the graphs to.")

# Enable the restriction of events to a particular regular expression.
flags.DEFINE_string(
    "task_name",
    ".*",
    "Regular expression that restricts the plots to the given tasks.",
)

# Execution modes for the script.
flags.DEFINE_bool("all", False, "Plot or show statistics for all graphs.")
flags.DEFINE_bool(
    "stats_only", False, "Print only statistics and skip output of plots."
)
flags.DEFINE_bool(
    "chrome_task_trace",
    False,
    "Outputs the given CSV files in the Chrome trace format in task-centric view.",
)
flags.DEFINE_bool(
    "chrome_resource_trace",
    False,
    "Outputs the given CSV files in the Chrome trace format in resource-centric view.",
)
flags.mark_bool_flags_as_mutual_exclusive(
    ["chrome_task_trace", "chrome_resource_trace"], required=False
)

# Enumerate the different kinds of plots.
flags.DEFINE_bool("plot_scheduler_runtime", False, "Plot scheduling runtime")
flags.DEFINE_string(
    "scheduler_runtime_timeline_plot_name",
    "scheduler_runtime_timeline.png",
    "The filename of the scheduler runtime timeline plot",
)
flags.DEFINE_string(
    "scheduler_runtime_cdf_plot_name",
    "scheduler_runtime_cdf.png",
    "The filename of the scheduler runtime CDF plot",
)

flags.DEFINE_bool("plot_utilization", False, "Plot resource utilization")
flags.DEFINE_string(
    "utilization_timeline_plot_name",
    "utilization_bar_chart.png",
    "The filename of the utilization timeline plot",
)

flags.DEFINE_bool("plot_task_placement_stats", False, "Plot task placement stats")
flags.DEFINE_string(
    "task_placement_bar_chart_plot_name",
    "tasks_placed_bar_chart.png",
    "The filename of the task placement bar char plot",
)

flags.DEFINE_bool("plot_task_slack", False, "Plot task slack")
flags.DEFINE_string(
    "task_slack_plot_name", "task_slack.png", "The filename of the task slack plot"
)

flags.DEFINE_bool("plot_task_placement_delay", False, "Plot task placement delay")
flags.DEFINE_string(
    "task_placement_delay_plot_name",
    "task_placement_delay.png",
    "The filename of the task start delay plot",
)

flags.DEFINE_bool(
    "plot_inter_task_time",
    False,
    "Plot histogram of time difference between task releases",
)
flags.DEFINE_string(
    "inter_task_time_plot_name",
    "inter_task_time.png",
    "The filename of the inter-task time plot.",
)

flags.DEFINE_bool(
    "plot_missed_deadlines",
    False,
    "Plots statistics about the missed deadlines from each invocation.",
)
flags.DEFINE_string(
    "missed_deadline_plot_name",
    "missed_deadline.png",
    "The filename of the missed deadline plot.",
)

flags.DEFINE_boolean(
    "plot_end_to_end_response_time",
    False,
    "Plots end to end response time for each timestamp.",
)
flags.DEFINE_string(
    "end_to_end_response_time_plot_name",
    "end_to_end_response_time.png",
    "The filename of the end-to-end response time plot.",
)

matplotlib.rcParams.update({"font.size": 16, "figure.autolayout": True})
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.family"] = "serif"
axes_fontsize = 16
markers = {"EDF": "o", "Gurobi": "x", "Z3": "+", "LSF": "^"}
colors = {"EDF": "r", "Gurobi": "b", "Z3": "y", "LSF": "b"}

logger = setup_logging("plotting")


def plot_utilization(
    csv_reader,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=True,
    figure_size=(14, 10),
    bar_width=1.0,
):
    """Plots the timeline of the utilization of different resources on the workers.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    # Plotting defaults.
    # hatches = ['//', '--', '**']
    # alphas = np.arange(0.2, 1.2, 0.2)
    resource_color = {"GPU": "red", "CPU": "green"}

    # Worker Pool statistics
    worker_pool_stats = csv_reader.get_worker_pool_utilizations(scheduler_csv_file)

    # Find all the resource types in the system.
    resource_types = set()
    for stats in worker_pool_stats:
        for resource in stats.resource_utilizations.keys():
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
    # resource_free_heights = {
    #     resource: [
    #         stat.resource_utilizations[resource][1] /
    #         sum(stat.resource_utilizations[resource])
    #         for stat in worker_pool_stats
    #     ]
    #     for resource in resource_types
    # }

    logger.debug("================== Resource Utilization ==================")
    for resource_type in resource_types:
        logger.debug("Utilization for {}".format(resource_type))
        log_statistics(resource_used_heights[resource_type], logger)

    # Plot a histogram with the results.
    if plot:
        fig = plt.figure(figsize=figure_size)
        # Keeping this for a little bit longer.
        # x_vals = np.arange(1, len(worker_pool_stats) + 1)
        # _ = [
        #     plt.bar(
        #         x_vals + bar_width * i,
        #         resource_used_heights[resource_type],
        #         width=bar_width,
        #         # edgecolor='black',
        #         # hatch=hatches[i],
        #         alpha=0.4,
        #         color=resource_color[resource_type],
        #     )
        #     for i, resource_type in enumerate(resource_types)
        # ]
        for i, resource_type in enumerate(resource_types):
            plt.scatter(
                sim_times_sec,
                resource_used_heights[resource_type],
                color=resource_color[resource_type],
            )

        plt.xlabel("Timeline [s]", fontsize=axes_fontsize)
        plt.ylabel("Normalized utilization", fontsize=axes_fontsize)
        # Set the y limits so we can visualize the legend.
        ax = fig.gca()
        ax.set_ylim(0, 1.01)
        plt.yticks([0, 0.5, 1.0])

        legend_elements = [
            Patch(
                facecolor=resource_color[resource_type],
                label=f"{scheduler_name}-{resource_type}",
            )
            for resource_type in resource_types
        ]
        plt.legend(handles=legend_elements, framealpha=0)
        plt.savefig(output, bbox_inches="tight")


def plot_scheduler_runtime(
    csv_reader,
    scheduler_csv_files,
    scheduler_labels,
    timeline_output,
    cdf_output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots a timeline and a CDF of the scheduler invocations.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_files (`Sequence[str]`): A list of paths to CSV files containing
            the results.
        scheduler_labels (`Sequence[str]`): A list of labels to give the schedulers.
        timeline_output (`str`): The path to where the timeline plot should be output.
        cdf_output (`str`): The path to where the CDF plot should be output.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    # Retrieve the runtime of the scheduler invocations.
    logger.debug("================= Scheduler runtime [ms] =================")
    max_start_time = -sys.maxsize
    max_runtime = -sys.maxsize
    all_runtimes = []
    all_start_times = []
    for csv_file in scheduler_csv_files:
        scheduler_invocations = csv_reader.get_scheduler_invocations(csv_file)
        runtimes = list(map(attrgetter("runtime"), scheduler_invocations))
        runtimes_ms = [runtime / 1000 for runtime in runtimes]
        start_times = list(map(attrgetter("start_time"), scheduler_invocations))
        start_times_ms = [start_time / 1000 for start_time in start_times]
        max_start_time = max(max_start_time, max(start_times_ms))
        max_runtime = max(max_runtime, max(runtimes_ms))
        all_runtimes.append(runtimes_ms)
        all_start_times.append(start_times_ms)
        logger.debug(f"Stats for {csv_file}")
        log_statistics(runtimes_ms, logger)

    # Plot a timelapse of the runtime of the scheduler.
    if plot:
        plt.figure(figsize=figure_size)
        for i, label in enumerate(scheduler_labels):
            plt.plot(
                all_start_times[i],
                all_runtimes[i],
                label=label,
                marker=markers[label],
                color=colors[label],
            )
        plt.xlim(0, max_start_time)
        plt.xlabel("Timeline [ms]", fontsize=axes_fontsize)
        plt.ylabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
        plt.legend(frameon=False)
        plt.savefig(timeline_output, bbox_inches="tight")

        # Plot the CDF of the runtime of the scheduler invocations.
        plt.figure(figsize=figure_size)
        for i, label in enumerate(scheduler_labels):
            count, bin_count = np.histogram(all_runtimes[i], bins=100)
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            plt.plot(bin_count[1:], cdf, label=label)
        plt.xlim(0, max_runtime)
        plt.xlabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
        plt.ylabel("CDF", fontsize=axes_fontsize)
        plt.legend(frameon=False)
        plt.savefig(cdf_output, bbox_inches="tight")


def plot_task_placement_stats(
    csv_reader,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots the number of placed and unplaced tasks by each scheduler invocation.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    scheduler_invocations = csv_reader.get_scheduler_invocations(scheduler_csv_file)

    # Calculate the heights of placed and unplaced tasks.
    placed_task_heights = [
        scheduler_invocation.placed_tasks
        for scheduler_invocation in scheduler_invocations
    ]
    unplaced_task_heights = [
        scheduler_invocation.unplaced_tasks
        for scheduler_invocation in scheduler_invocations
    ]

    logger.debug("================== Num placed tasks ==================")
    log_statistics(placed_task_heights, logger)
    logger.debug("================== Num unplaced tasks ==================")
    log_statistics(unplaced_task_heights, logger)

    if plot:
        # Plot a histogram with the results.
        plt.figure(figsize=figure_size)
        x_vals = np.arange(1, len(scheduler_invocations) + 1)

        _ = plt.bar(
            x_vals,
            placed_task_heights,
            # edgecolor='black',
            # hatch='//',
            alpha=0.4,
            color="green",
        )
        _ = plt.bar(
            x_vals,
            unplaced_task_heights,
            bottom=placed_task_heights,
            # edgecolor='black',
            # hatch='**',
            alpha=0.4,
            color="red",
        )

        # Add the axis labels.
        plt.xlabel("Scheduler Run", fontsize=axes_fontsize)
        plt.ylabel("Placed and Unplaced Tasks", fontsize=axes_fontsize)

        # Add the yticks
        max_y_val = max(
            scheduler_invocation.total_tasks
            for scheduler_invocation in scheduler_invocations
        )
        plt.ylim(0, int(max_y_val))

        # Set the legend.
        legend_elements = [
            Patch(facecolor="green", label=f"{scheduler_name} Placed Tasks"),
            Patch(facecolor="red", label=f"{scheduler_name} Unplaced Tasks"),
        ]
        plt.legend(handles=legend_elements, framealpha=0)
        plt.savefig(output, bbox_inches="tight")


def plot_inter_task_time(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots the inter-task time for tasks that match the given regex.

    The inter-task time is the actual time elapsed between the release time of two
    successive invocations of a particular task.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    # Retrieve the tasks from the CSV file that match the given regular expression.
    tasks = csv_reader.get_tasks(scheduler_csv_file)
    task_map = defaultdict(list)
    for task in tasks:
        if re.match(task_name_regex, task.name):
            task_map[task.name].append(task)

    # Collect the inter-task time statistics and print them.
    logger.debug("================== Inter-Task Time [ms] ==================")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    inter_release_times = []
    labels = []
    for task_name, tasks in task_map.items():
        sorted_tasks = sorted(tasks)
        last_release = sorted_tasks[0].release_time
        task_inter_release_time = []
        for index in range(1, len(tasks)):
            task_inter_release_time.append(
                (sorted_tasks[index].release_time - last_release) / 1000
            )
            last_release = sorted_tasks[index].release_time
        labels.append(task_name)
        inter_release_times.append(task_inter_release_time)
        logger.debug(f"Statistics for {task_name}:")
        log_statistics(task_inter_release_time, logger, offset="      ")

    if plot:
        plt.figure(figsize=figure_size)
        plt.xlabel("Inter-Task Time [ms]", fontsize=axes_fontsize)
        plt.hist(inter_release_times, label=labels, density=False, bins=100)
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def plot_task_slack(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots the intended and actual slack of the tasks.

    The actual slack is defined as the time between the deadline and the actual
    completion of the task, where a negative value defines a missed deadline.

    The intended slack is defined as the time between the deadline and the intended
    finish time of the task (as defined by its intended start time and runtime).

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    # Retrieve the tasks that match the given regular expression.
    tasks = []
    for task in csv_reader.get_tasks(scheduler_csv_file):
        if re.match(task_name_regex, task.name):
            tasks.append(task)

    # Compute the time between the deadline and the actual completion of the task.
    slack = [(task.deadline - task.completion_time) / 1000 for task in tasks]
    logger.debug("================== Task completion slack [ms] ==================")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    log_statistics(slack, logger)

    # Compute the time between the deadline and the intended completion of the task.
    initial_slack = [
        (task.deadline - task.release_time - task.runtime) / 1000 for task in tasks
    ]
    logger.debug("================== Task placement slack [ms] ==================")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    log_statistics(initial_slack, logger)

    # If required, plot a histogram of the slack from the deadline for the tasks.
    if plot:
        plt.figure(figsize=figure_size)
        plt.xlim(min(slack), max(slack))
        plt.xlabel("Task Slack [ms]", fontsize=axes_fontsize)
        plt.ylabel("Number of Tasks", fontsize=axes_fontsize)
        plt.hist(
            [slack, initial_slack],
            label=[
                f"{scheduler_name} Task Completion Slack",
                f"{scheduler_name} Task Placement Slack",
            ],
            density=False,
            bins=100,
        )
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def plot_task_placement_delay(
    csv_reader,
    scheduler_csv_files,
    scheduler_labels,
    output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots a histogram of the placement delays.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_files (`Sequence[str]`): A list of paths to CSV files containing
            the results.
        scheduler_labels (`Sequence[str]`): A list of labels to give the schedulers.
        output (`str`): The path to where the plot should be output.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    logger.debug("================ Task placement delay [ms] ================")
    placement_delays = []
    min_delay = sys.maxsize
    max_delay = -sys.maxsize
    plot_colors = [colors[label] for label in scheduler_labels]
    for csv_file in scheduler_csv_files:
        task_placements = csv_reader.get_task_placements(csv_file)
        placement_delay = [
            (placement.simulator_time - placement.task.release_time) / 1000
            for placement in task_placements
        ]
        placement_delays.append(placement_delay)
        min_delay = min(min_delay, min(placement_delay))
        max_delay = max(max_delay, max(placement_delay))
        logger.debug(f"Placement delay stats for {csv_file}")
        log_statistics(placement_delay, logger)

    if plot:
        plt.figure(figsize=figure_size)
        plt.xlim(min_delay, max_delay)
        plt.xlabel("Task Placement Delay [ms]", fontsize=axes_fontsize)
        plt.ylabel("Number of Tasks", fontsize=axes_fontsize)
        plt.hist(
            placement_delays,
            density=False,
            bins=100,
            label=scheduler_labels,
            color=plot_colors,
        )
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def task_stats(tasks):
    for task in tasks:
        logger.debug(f"Task Name: {task.name}\n\tRelease Time: {task.release_time}\n\t")
        logger.debug(
            f"Completion Time: {task.completion_time}\n\tDeadline: {task.deadline}\n\t"
        )
        logger.debug(
            f"Runtime: {task.runtime}\n\tSlack: {task.deadline - task.completion_time}"
        )


def plot_missed_deadlines(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=True,
    figure_size=(14, 10),
):
    """Plots the number of missed deadlines by each task.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
    """
    # Group the missed deadlines by their task name (if regex is matched).
    missed_deadlines = csv_reader.get_missed_deadline_events(scheduler_csv_file)
    missed_deadline_by_task_name = defaultdict(list)
    missed_deadline_delays = []
    for _, task in missed_deadlines:
        if re.match(task_name_regex, task.name):
            missed_deadline_by_task_name[task.name].append(task)
            missed_deadline_delays.append((task.completion_time - task.deadline) / 1000)

    # Log the results.
    logger.debug("==================  Missed Deadlines ==================")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    logger.debug(f"Average missed deadline delay: {np.mean(missed_deadline_delays)}")
    for task_name, tasks in missed_deadline_by_task_name.items():
        logger.debug(f"{task_name}: {len(tasks)}")
        missed_deadline_delays_per_task = [
            (task.completion_time - task.deadline) / 1000 for task in tasks
        ]
        logger.debug(f"    Average: {np.mean(missed_deadline_delays_per_task)}")

    if plot:
        # Plot the number of missed deadlines by the method name.
        plt.figure(figsize=figure_size)
        plt.bar(
            missed_deadline_by_task_name.keys(),
            list(map(len, missed_deadline_by_task_name.values())),
        )
        plt.xticks(rotation=90)
        plt.savefig(output, bbox_inches="tight")


def plot_end_to_end_response_time(
    csv_reader,
    scheduler_csv_file,
    output,
    plot=True,
    figure_size=(14, 10),
):
    tasks = csv_reader.get_tasks(scheduler_csv_file)
    timestamp_start_end = {}
    for task in tasks:
        if task.timestamp in timestamp_start_end:
            release_time = min(
                task.release_time, timestamp_start_end[task.timestamp][0]
            )
            completion_time = max(
                task.completion_time, timestamp_start_end[task.timestamp][1]
            )
            timestamp_start_end[task.timestamp] = (release_time, completion_time)
        else:
            timestamp_start_end[task.timestamp] = (
                task.release_time,
                task.completion_time,
            )

    e2e_response_time = [ct - rt for (rt, ct) in timestamp_start_end.values()]
    logger.debug("================== End-to-end response time [ms] ==================")
    log_statistics(e2e_response_time, logger)

    if plot:
        plt.figure(figsize=figure_size)
        plt.xlabel("End-to-end response time [ms]", fontsize=axes_fontsize)
        plt.hist(e2e_response_time, density=False, bins=100)
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def log_basic_task_statistics(logger, csv_reader: CSVReader, csv_file: str):
    """Prints the basic task statistics from the given CSV file.

    Args:
        logger (logging.Logger): The logger instance to show results on.
        csv_reader (:py:class:`CSVReader`): The CSVReader instance containing the
            results.
        csv_file (str): The path to the CSV file to show the results for.
    """
    # Get the tasks grouped by their name.
    tasks = defaultdict(list)
    for task in csv_reader.get_tasks(csv_file):
        tasks[task.name].append(task)

    # Get the placements grouped by the task name.
    placements = defaultdict(list)
    for placement in csv_reader.get_task_placements(csv_file):
        placements[placement.task.name].append(placement)

    # Gather the results.
    results = []
    total_missed_deadline_delays = []
    total_placement_delays = []
    for task_name, grouped_tasks in tasks.items():
        # Gather missed deadline delays.
        missed_deadline_tasks = [task for task in grouped_tasks if task.missed_deadline]
        missed_deadline_delays = [
            (task.completion_time - task.deadline) / 1000
            for task in missed_deadline_tasks
        ]
        total_missed_deadline_delays.extend(missed_deadline_delays)

        # Gather placement delays.
        placement_delays = [
            (placement.simulator_time - placement.task.release_time) / 1000
            for placement in placements[task_name]
        ]
        total_placement_delays.extend(placement_delays)

        results.append(
            (
                task_name,
                len(grouped_tasks),
                len(missed_deadline_tasks),
                np.mean(missed_deadline_delays)
                if len(missed_deadline_delays) != 0
                else 0.0,
                np.mean(placement_delays),
            )
        )
    results.append(
        (
            "Total",
            len(csv_reader.get_tasks(csv_file)),
            len(csv_reader.get_missed_deadline_events(csv_file)),
            np.mean(total_missed_deadline_delays)
            if len(total_missed_deadline_delays) != 0
            else 0.0,
            np.mean(total_placement_delays),
        )
    )

    # Log the results.
    logger.debug(
        "\n"
        + tabulate(
            results,
            headers=[
                "Name",
                "Total",
                "X Dline?",
                "Dline Delay",
                "Place Delay",
            ],
            tablefmt="grid",
            showindex=True,
        )
    )


def main(argv):
    assert len(FLAGS.csv_files) == len(
        FLAGS.csv_labels
    ), "Mismatch between length of csv files and labels flags."

    try:
        re.compile(FLAGS.task_name)
    except re.error:
        raise ValueError(
            f"The regular expression for Task names: {FLAGS.task_name} is invalid."
        )

    figure_size = (14, 10)

    # Load the events from the CSV file into the CSVReader class.
    csv_reader = CSVReader(csv_paths=FLAGS.csv_files)

    for scheduler_csv_file, scheduler_label in zip(FLAGS.csv_files, FLAGS.csv_labels):
        # Log basic statistics
        simulation_end_time = csv_reader.get_simulator_end_time(scheduler_csv_file)
        logger.debug(
            f"Simulation end time for {scheduler_csv_file}: {simulation_end_time}"
        )
        log_basic_task_statistics(logger, csv_reader, scheduler_csv_file)

        # Output the Chrome trace format if requested.
        if FLAGS.chrome_task_trace or FLAGS.chrome_resource_trace:
            filename = Path(scheduler_csv_file).stem
            output_path = os.path.join(FLAGS.output_dir, filename + ".json")
            logger.debug(f"Saving trace for {scheduler_csv_file} at {output_path}")
            csv_reader.to_chrome_trace(
                scheduler_csv_file,
                scheduler_label,
                output_path,
                task_centric=FLAGS.chrome_task_trace,
            )

        # Plot the requested graphs.
        if FLAGS.plot_utilization or FLAGS.all:
            plot_utilization(
                csv_reader,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.utilization_timeline_plot_name}",
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )
        if FLAGS.plot_task_placement_stats or FLAGS.all:
            plot_task_placement_stats(
                csv_reader,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.task_placement_bar_chart_plot_name}",
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )
        if FLAGS.plot_task_slack or FLAGS.all:
            plot_task_slack(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir, f"{scheduler_label}_{FLAGS.task_slack_plot_name}"
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )
        if FLAGS.plot_inter_task_time or FLAGS.all:
            plot_inter_task_time(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.inter_task_time_plot_name}",
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )
        if FLAGS.plot_missed_deadlines or FLAGS.all:
            plot_missed_deadlines(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.missed_deadline_plot_name}",
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )
        if FLAGS.plot_end_to_end_response_time or FLAGS.all:
            plot_end_to_end_response_time(
                csv_reader,
                scheduler_csv_file,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.end_to_end_response_time_plot_name}",
                ),
                plot=not FLAGS.stats_only,
                figure_size=figure_size,
            )

    if FLAGS.plot_scheduler_runtime or FLAGS.all:
        plot_scheduler_runtime(
            csv_reader,
            FLAGS.csv_files,
            FLAGS.csv_labels,
            os.path.join(FLAGS.output_dir, FLAGS.scheduler_runtime_timeline_plot_name),
            os.path.join(FLAGS.output_dir, FLAGS.scheduler_runtime_cdf_plot_name),
            plot=not FLAGS.stats_only,
            figure_size=figure_size,
        )
    if FLAGS.plot_task_placement_delay or FLAGS.all:
        plot_task_placement_delay(
            csv_reader,
            FLAGS.csv_files,
            FLAGS.csv_labels,
            os.path.join(FLAGS.output_dir, FLAGS.task_placement_delay_plot_name),
            plot=not FLAGS.stats_only,
            figure_size=figure_size,
        )


if __name__ == "__main__":
    app.run(main)
