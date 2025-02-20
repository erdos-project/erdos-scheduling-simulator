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
from main import FLAGS as main_flags
from utils import STATS_FUNCTIONS, log_statistics, setup_logging

# The formats that the Chrome trace can be output to.
TRACE_FORMATS = ("task", "taskgraph", "application", "resource", "scheduler")

# The formats that the task stats can be output to.
TASK_STATS_FORMATS = ("basic", "detailed")

FLAGS = flags.FLAGS

# Required inputs for either (csv_files, label) or (csv_files, conf_files)
flags.DEFINE_list("csv_files", None, "List of CSV files containing experiment logs")
flags.mark_flag_as_required("csv_files")
flags.register_validator(
    "csv_files",
    lambda value: all(f.endswith(".csv") for f in value),
    message="All files must end with .csv extension",
)

flags.DEFINE_list("csv_labels", None, "List of labels to use for the experiment logs")
flags.register_validator(
    "csv_labels",
    lambda value: len(value) == len(FLAGS.csv_files) if value else True,
    message="Mismatch between length of csv files and labels flags.",
)

flags.DEFINE_list("conf_files", None, "List of conf files used to run the experiments.")
flags.register_validator(
    "conf_files",
    lambda value: (len(value) == len(FLAGS.csv_files) if value else True)
    and (all(f.endswith(".conf") for f in value) if value else True),
    message="All files must end with .conf extension.",
)

flags.register_multi_flags_validator(
    ["csv_labels", "conf_files"],
    lambda values: values["csv_labels"] is not None or values["conf_files"] is not None,
    message="Atleast one of csv_labels or conf_files must be provided.",
)

flags.DEFINE_string("output_dir", ".", "The directory to output the graphs to.")

# Enable the restriction of events to a particular regular expression or time.
flags.DEFINE_string(
    "task_name",
    ".*",
    "Regular expression that restricts the plots to the given tasks.",
)
flags.DEFINE_list(
    "between_time",
    None,
    "The time for which to show the results (for Chrome traces) [in microseconds]",
)
flags.register_validator(
    "between_time",
    lambda value: FLAGS.chrome_trace is not None if value else True,
    message="The between_time flag can only be used for Chrome traces.",
)
flags.register_validator(
    "between_time",
    lambda value: len(value) <= 2 if value else True,
    message="The between_time flag can have a maximum of two values.",
)

# Execution modes for the script.
flags.DEFINE_bool("all", False, "Plot or show statistics for all graphs.")
flags.DEFINE_bool(
    "plot", False, "Plots the graph in addition to showing the statistics."
)
flags.DEFINE_string(
    "chrome_trace",
    None,
    f"Outputs the CSV data as a Chrome trace. Allowed values: {TRACE_FORMATS}",
)
flags.register_validator(
    "chrome_trace",
    lambda value: value in TRACE_FORMATS if value is not None else True,
    message=f"Only {TRACE_FORMATS} Chrome trace formats are allowed.",
)

# Allow the choice of seeing release times, deadlines and tasks with placement issues
# in Chrome trace.
flags.DEFINE_enum(
    "show_deadlines",
    None,
    ["never", "missed", "always"],
    "Chooses if the deadline events are shown in the Chrome trace. "
    "'never' shows no deadline events, 'missed' shows only deadlines that were missed, "
    "and 'always' shows all deadlines.",
)
flags.register_validator(
    "show_deadlines",
    lambda value: FLAGS.chrome_trace is not None if value else True,
    message="The show_deadlines flag can only be used for Chrome traces.",
)

flags.DEFINE_enum(
    "show_release_times",
    None,
    ["never", "intended", "always"],
    "Chooses if the release time events are shown in the Chrome trace. "
    "'never' shows no release time events, 'intended' shows only intended release "
    "times that were set by the Simulator, and 'always' shows the actual release "
    "times for all the tasks.",
)
flags.register_validator(
    "show_release_times",
    lambda value: FLAGS.chrome_trace is not None if value else True,
    message="The show_release_times flag can only be used for Chrome traces.",
)

flags.DEFINE_boolean(
    "with_placement_issues",
    False,
    "Show the tasks with the placement issues between the requested times.",
)
flags.register_validator(
    "with_placement_issues",
    lambda value: (
        (FLAGS.chrome_trace == "resource" and FLAGS.between_time is not None)
        if value
        else True
    ),
    message="Placement issues require resource traces with limited interval.",
)

# Allow the choice of statistics to show for the metrics.
flags.DEFINE_list("stat", "all", "The statistic to show for the given metric.")
flags.register_validator(
    "stat",
    lambda value: all(s in STATS_FUNCTIONS or s == "all" for s in value),
    message=f"Only {tuple(STATS_FUNCTIONS.keys())} statistic functions are allowed.",
)

# Enumerate the different kinds of metrics.
flags.DEFINE_bool(
    "sched_runtime", False, "Analyze the runtime of the scheduler invocations."
)
flags.DEFINE_string(
    "sched_runtime_timeline_plot_name",
    "sched_runtime_timeline.png",
    "The filename of the scheduler runtime timeline plot",
)
flags.DEFINE_string(
    "sched_runtime_cdf_plot_name",
    "sched_runtime_cdf.png",
    "The filename of the scheduler runtime CDF plot",
)

flags.DEFINE_bool(
    "resource_utilization", False, "Analyze the resource utilization of the trace."
)
flags.DEFINE_string(
    "resource_utilization_timeline_plot_name",
    "resource_utilization_bar_chart.png",
    "The filename of the resource utilization timeline plot",
)

flags.DEFINE_bool(
    "task_placement",
    False,
    "Analyzes the number of placed and unplaced tasks in the trace.",
)
flags.DEFINE_string(
    "task_placement_bar_chart_plot_name",
    "tasks_placed_bar_chart.png",
    "The filename of the task placement bar char plot",
)

flags.DEFINE_bool(
    "task_slack",
    False,
    "Analyzes the placement and completion slack of the tasks in the trace.",
)
flags.DEFINE_string(
    "task_slack_plot_name", "task_slack.png", "The filename of the task slack plot"
)

flags.DEFINE_bool("task_placement_delay", False, "Plot task placement delay")
flags.DEFINE_string(
    "task_placement_delay_plot_name",
    "task_placement_delay.png",
    "The filename of the task start delay plot",
)

flags.DEFINE_bool(
    "inter_task_time",
    False,
    "Analyzes the time difference between task releases.",
)
flags.DEFINE_string(
    "inter_task_time_plot_name",
    "inter_task_time.png",
    "The filename of the inter-task time plot.",
)

flags.DEFINE_bool(
    "missed_deadlines",
    False,
    "Analyzes the statistics about the missed deadlines from each invocation.",
)
flags.DEFINE_bool(
    "plot_goodput_graph",
    False,
    "Plot the goodput of the scheduler runs",
)

flags.DEFINE_string(
    "goodput_plot_name",
    "goodput.png",
    "The filename of the goodput plot.",
)
flags.DEFINE_string(
    "goodput_plot_title",
    "Goodput Comparison",
    "The title of the goodput plot.",
)

flags.DEFINE_string(
    "missed_deadline_plot_name",
    "missed_deadline.png",
    "The filename of the missed deadline plot.",
)

flags.DEFINE_bool(
    "goodresource_utilization",
    False,
    "Plot a timeline of the good resource utilization. "
    "A good resource utilization is defined as the utilization of the resource "
    "for the use of a TaskGraph that meets its deadline.",
)
flags.DEFINE_string(
    "goodresource_utilization_plot_name",
    "goodresource_utilization.png",
    "The filename of the good resource utilization plot.",
)

flags.DEFINE_bool(
    "end_to_end_response_time",
    False,
    "Analyzes the end-to-end response time for each timestamp.",
)
flags.DEFINE_string(
    "end_to_end_response_time_plot_name",
    "end_to_end_response_time.png",
    "The filename of the end-to-end response time plot.",
)

flags.DEFINE_string(
    "task_stats",
    None,
    "Log basic or detailed statistics for each task grouped by the task name.",
)
flags.register_validator(
    "task_stats",
    lambda value: value in TASK_STATS_FORMATS if value is not None else True,
    message=f"Only {TASK_STATS_FORMATS} formats are allowed for task statistics.",
)

flags.DEFINE_bool(
    "aggregate_stats",
    False,
    "Log aggregate stats from a list of (csv_files, conf_files)",
)
flags.register_multi_flags_validator(
    ["aggregate_stats", "conf_files"],
    lambda values: (
        values["conf_files"] is not None if values["aggregate_stats"] else True
    ),
    message="Config files must be provided to invoke the aggregate_stats mode.",
)

matplotlib.rcParams.update({"font.size": 16, "figure.autolayout": True})
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.family"] = "serif"
axes_fontsize = 16
markers = {"EDF": "o", "Gurobi": "x", "Z3": "+", "LSF": "^"}
colors = {"EDF": "r", "Gurobi": "b", "Z3": "y", "LSF": "b"}

logger = setup_logging("plotting")


def __parse_flagfile(path: str):
    """Parses the flagfile at the given path, and sets the FLAGS to contain the new
    values.

    Args:
        path (str): The path to where the config file is stored.
    """
    main_flags([__file__, f"--flagfile={path}"])


def analyze_resource_utilization(
    csv_reader,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=False,
    figure_size=(14, 10),
    bar_width=1.0,
    stats="all",
):
    """Analyzes the utilization of different resources on the workers.

    If `plot` is `True`, the method plots a timeline of the utilization of different
    resources on the workers.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
    # Plotting defaults.
    # hatches = ['//', '--', '**']
    # alphas = np.arange(0.2, 1.2, 0.2)
    resource_color = {"Slot": "green"}

    # Worker Pool statistics
    worker_pool_stats = csv_reader.get_worker_pool_utilizations(scheduler_csv_file)

    # Find all the resource types in the system.
    resource_types = set()
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
        log_statistics(resource_used_heights[resource_type], logger, stats)

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

        plt.xlim(0, max(sim_times_sec))
        plt.xlabel("Timeline [s]", fontsize=axes_fontsize)
        plt.ylabel("Normalized Utilization", fontsize=axes_fontsize)
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


def analyze_scheduler_runtime(
    csv_reader,
    scheduler_csv_files,
    scheduler_labels,
    timeline_output,
    cdf_output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the runtime of the scheduler invocations from the CSV files.

    If `plot` is `True`, the method plots a timeline and a CDF of the runtime of the
    scheduler invocations.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_files (`Sequence[str]`): A list of paths to CSV files containing
            the results.
        scheduler_labels (`Sequence[str]`): A list of labels to give the schedulers.
        timeline_output (`str`): The path to where the timeline plot should be output.
        cdf_output (`str`): The path to where the CDF plot should be output.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
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
        log_statistics(runtimes_ms, logger, stats)

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
        plt.ylim(0, max(50, max([max(a_r) for a_r in all_runtimes])))
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
        plt.ylim(0, 1)
        plt.xlabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
        plt.ylabel("CDF", fontsize=axes_fontsize)
        plt.legend(frameon=False)
        plt.savefig(cdf_output, bbox_inches="tight")


def analyze_task_placement(
    csv_reader: CSVReader,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the number of placed and unplaced tasks by each scheduler invocation.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
    scheduler_invocations = csv_reader.get_scheduler_invocations(scheduler_csv_file)

    # Calculate the heights of placed and unplaced tasks.
    placed_task_heights = [
        scheduler_invocation.num_placed_tasks
        for scheduler_invocation in scheduler_invocations
    ]
    unplaced_task_heights = [
        scheduler_invocation.num_unplaced_tasks
        for scheduler_invocation in scheduler_invocations
    ]

    logger.debug("================== Num placed tasks ==================")
    log_statistics(placed_task_heights, logger, stats)
    logger.debug("================== Num unplaced tasks ==================")
    log_statistics(unplaced_task_heights, logger, stats)

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

        plt.xlim(0, max(x_vals))

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


def analyze_inter_task_time(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the inter-task time for tasks that match the given regex.

    The inter-task time is the actual time elapsed between the release time of two
    successive invocations of a particular task.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
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
        log_statistics(task_inter_release_time, logger, stats, offset="      ")

    if plot:
        plt.figure(figsize=figure_size)
        plt.xlabel("Inter-Task Time [ms]", fontsize=axes_fontsize)
        plt.hist(inter_release_times, label=labels, density=False, bins=100)
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def analyze_task_slack(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the actual and intended completion slack of the tasks.

    The actual completion slack is defined as the time between the deadline and the
    actual completion of the task, where a negative value defines a missed deadline.

    The intended completion slack is defined as the time between the deadline and the
    intended finish time of the task (as defined by its intended start time and
    runtime).

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
    # Retrieve the tasks that match the given regular expression.
    tasks = []
    for task in csv_reader.get_tasks(scheduler_csv_file):
        if re.match(task_name_regex, task.name):
            tasks.append(task)

    # Compute the time between the deadline and the actual completion of the task.
    slack = [-task.get_deadline_delay() / 1000 for task in tasks]
    logger.debug("================== Actual task completion slack [ms] ===============")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    log_statistics(slack, logger, stats)

    # Compute the time between the deadline and the intended completion of the task.
    initial_slack = [
        (task.deadline - task.release_time - task.runtime) / 1000 for task in tasks
    ]
    logger.debug("================== Intended task completion slack [ms] =============")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    log_statistics(initial_slack, logger, stats)

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


def analyze_task_placement_delay(
    csv_reader,
    scheduler_csv_files,
    scheduler_labels,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the placement delays from the given trace.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_files (`Sequence[str]`): A list of paths to CSV files containing
            the results.
        scheduler_labels (`Sequence[str]`): A list of labels to give the schedulers.
        output (`str`): The path to where the plot should be output.
        plot (`bool`) [default = True]: Show statistics only if set to False.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
    logger.debug("================ Task placement delay [ms] ================")
    placement_delays = []
    min_delay = sys.maxsize
    max_delay = -sys.maxsize
    plot_colors = [colors[label] for label in scheduler_labels]
    for csv_file in scheduler_csv_files:
        placement_delay = [
            task.get_placement_delay() / 1000 for task in csv_reader.get_tasks(csv_file)
        ]
        placement_delays.append(placement_delay)
        min_delay = min(min_delay, min(placement_delay))
        max_delay = max(max_delay, max(placement_delay))
        logger.debug(f"Placement delay stats for {csv_file}")
        log_statistics(placement_delay, logger, stats)

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


def log_detailed_task_statistics(
    task_name_regex: str, csv_reader: CSVReader, csv_file: str
):
    """Prints the detailed statistics for the given CSV file.

    Args:
        task_name_regex (`str`): The regular expression to match the task name to.
        csv_reader (:py:class:`CSVReader`): The CSVReader instance containing the
            results.
        csv_file (str): The path to the CSV file to show the results for.
    """
    # Get the tasks grouped by their name.
    grouped_tasks = defaultdict(list)
    for task in csv_reader.get_tasks(csv_file):
        if re.match(task_name_regex, task.name):
            grouped_tasks[task.name].append(task)

    # Colorify the results.
    R = "\033[0;31;40m"
    G = "\033[0;32;40m"
    D = "\033[0m"

    # Log the results.
    for task_name, tasks in grouped_tasks.items():
        results = []
        for task in tasks:
            results.append(
                tuple(
                    map(
                        lambda val: (
                            R + str(val) + D
                            if task.missed_deadline
                            else G + str(val) + D
                        ),
                        (
                            task.timestamp,
                            str(task.id),
                            (
                                task.intended_release_time / 1000
                                if task.intended_release_time != -1
                                else "-"
                            ),
                            task.release_time / 1000,
                            task.placement_time / 1000,
                            task.start_time / 1000,
                            task.runtime / 1000,
                            task.deadline / 1000,
                            task.completion_time / 1000,
                            task.missed_deadline,
                            len(task.skipped_times) > 0,
                            task.get_deadline_delay() / 1000,
                            task.get_placement_delay() / 1000,
                            (
                                task.get_release_delay() / 1000
                                if task.intended_release_time != -1
                                else "-"
                            ),
                        ),
                    )
                )
            )
        logger.debug(
            f"Detailed view for {task_name}:\n"
            + tabulate(
                results,
                headers=[
                    "Time",
                    "ID",
                    "Int Release",
                    "Release",
                    "Placement",
                    "Start",
                    "Runtime",
                    "Deadline",
                    "Completion",
                    "X Dline?",
                    "X Skipped",
                    "Dline Delay",
                    "Place Delay",
                    "Rels Delay",
                ],
                tablefmt="grid",
                showindex=True,
            )
        )


def analyze_missed_deadlines(
    csv_reader,
    task_name_regex,
    scheduler_csv_file,
    scheduler_name,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the number of missed deadlines by each task.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        task_name_regex (`str`): The regular expression to match the task name to.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        scheduler_name (`str`): The label to give to the scheduler.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
    # Group the missed deadlines by their task name (if regex is matched).
    missed_deadline_by_task_name = defaultdict(list)
    missed_deadline_delays = []
    for task in csv_reader.get_tasks(scheduler_csv_file):
        if re.match(task_name_regex, task.name) and task.missed_deadline:
            missed_deadline_by_task_name[task.name].append(task)
            missed_deadline_delays.append(task.get_deadline_delay() / 1000)

    # Log the results.
    logger.debug("==================  Missed Deadlines ==================")
    logger.debug(f"Tasks that match the regex: {task_name_regex}")
    logger.debug(f"Average missed deadline delay: {np.mean(missed_deadline_delays)}")
    for task_name, tasks in missed_deadline_by_task_name.items():
        logger.debug(f"{task_name}")
        missed_deadline_delays_per_task = [
            task.get_deadline_delay() / 1000 for task in tasks
        ]
        log_statistics(missed_deadline_delays_per_task, logger, stats, offset="    ")

    if plot:
        # Plot the number of missed deadlines by the method name.
        plt.figure(figsize=figure_size)
        plt.bar(
            missed_deadline_by_task_name.keys(),
            list(map(len, missed_deadline_by_task_name.values())),
        )
        plt.xticks(rotation=90)
        plt.ylabel("Number of Missed Deadlines")
        plt.savefig(output, bbox_inches="tight")


def analyze_end_to_end_response_time(
    csv_reader,
    scheduler_csv_file,
    scheduler_label,
    output,
    plot=False,
    figure_size=(14, 10),
    stats="all",
):
    """Analyzes the end-to-end response time of each timestamp in the trace.

    Args:
        csv_reader (`data.CSVReader`): The CSVReader instance containing the parsed
            CSV file.
        scheduler_csv_file (`str`): The path to the CSV file for which to plot results.
        output (`str`): The path to where the plot should be output to.
        plot (`bool`) [default = False]: Plots the graphs if set to True.
        figure_size (`Tuple[int, int]`) [default=(14, 10)]: The size of the plot.
        stats (`Union[str, Sequence[str]`): The statistics to show for the metric.
    """
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

    e2e_response_time = [(ct - rt) / 1000 for (rt, ct) in timestamp_start_end.values()]
    logger.debug("================== End-to-end response time [ms] ==================")
    log_statistics(e2e_response_time, logger, stats)

    if plot:
        plt.figure(figsize=figure_size)
        plt.xlim(0, max(600, max(e2e_response_time)))
        plt.xlabel("End-to-end response time [ms]", fontsize=axes_fontsize)
        plt.hist(
            e2e_response_time,
            label=scheduler_label,
            density=False,
            range=(0, max(e2e_response_time)),
            bins=100,
        )
        plt.legend(frameon=False)
        plt.savefig(output, bbox_inches="tight")


def log_basic_task_statistics(
    task_name_regex, csv_reader: CSVReader, csv_file: str, stat: str = "p50"
):
    """Prints the basic task statistics from the given CSV file.

    Args:
        task_name_regex (`str`): The regular expression to match the task name to.
        csv_reader (:py:class:`CSVReader`): The CSVReader instance containing the
            results.
        csv_file (str): The path to the CSV file to show the results for.
        stat (str): The stat to show in the table for the given tasks.
    """
    # Get the Statistic function to use.
    if stat not in STATS_FUNCTIONS:
        raise ValueError(
            f"Requested stat: {stat} not found in {STATS_FUNCTIONS.keys()}"
        )
    stat_function = STATS_FUNCTIONS[stat][0]

    # Get the tasks grouped by their name.
    tasks = defaultdict(list)
    for task in csv_reader.get_tasks(csv_file):
        if re.match(task_name_regex, task.name):
            tasks[task.name].append(task)

    # Gather the results.
    results = []
    total_release_delays = []
    total_placement_delays = []
    total_missed_deadline_delays = []
    total_skipped = []
    for task_name, grouped_tasks in tasks.items():
        # Gather the number of skipped tasks.
        skipped_tasks = list(
            filter(lambda task: len(task.skipped_times) > 0, grouped_tasks)
        )
        total_skipped.extend(skipped_tasks)

        # Gather release delays.
        release_delays = [task.get_release_delay() / 1000 for task in grouped_tasks]
        total_release_delays.extend(release_delays)

        # Gather placement delays.
        placement_delays = [task.get_placement_delay() / 1000 for task in grouped_tasks]
        total_placement_delays.extend(placement_delays)

        # Gather missed deadline delays.
        missed_deadline_delays = [
            task.get_deadline_delay() / 1000
            for task in filter(lambda task: task.missed_deadline, grouped_tasks)
        ]
        total_missed_deadline_delays.extend(missed_deadline_delays)

        results.append(
            (
                task_name,
                len(grouped_tasks),
                len(missed_deadline_delays),
                len(skipped_tasks),
                stat_function(release_delays),
                stat_function(placement_delays),
                (
                    stat_function(missed_deadline_delays)
                    if len(missed_deadline_delays) != 0
                    else 0.0
                ),
            )
        )
    results.append(
        (
            "Total",
            sum(map(len, tasks.values())),
            len(total_missed_deadline_delays),
            len(total_skipped),
            stat_function(total_release_delays),
            stat_function(total_placement_delays),
            (
                stat_function(total_missed_deadline_delays)
                if len(total_missed_deadline_delays) != 0
                else 0.0
            ),
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
                "# Skipped",
                "Rls Delay",
                "Place Delay",
                "Dline Delay",
            ],
            tablefmt="grid",
            showindex=True,
        )
    )


def plot_goodput(
    csv_reader, csv_files, scheduler_labels, output, figure_size=(14, 10), title=""
):
    x, y = [], []
    for csv_file, scheduler_label in zip(csv_files, scheduler_labels):
        goodput = csv_reader.get_goodput(csv_file)
        x.append(scheduler_label)
        y.append(goodput)
        logger.info(f"Scheduler: {scheduler_label} goodput: {goodput}")
    plt.figure(figsize=figure_size)
    plt.bar(x, y)
    plt.xlabel("Schedulers")
    plt.ylabel("Goodput")
    xlocs, _ = plt.xticks()
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.title(title)
    plt.savefig(output, bbox_inches="tight")


def plot_goodresource_utilization(
    csv_reader,
    csv_files,
    scheduler_labels,
    plot=False,
    output_dir="./",
    output_file_name="goodresource_utilization.png",
):
    """Plots the timeline of the resource utilization.

    A good resource utilization is defined as the utilization of the resource for the
    purposes of meeting the deadline of a TaskGraph."""
    # Retrieve the Tasks that were released into the system.
    for csv_file, scheduler_label in zip(csv_files, scheduler_labels):
        tasks = csv_reader.get_tasks(csv_file)
        task_graphs = csv_reader.get_task_graph(csv_file)
        resource_utilization = {}
        simulator_end_time = 0
        for task in tasks:
            if task.task_graph not in task_graphs:
                raise ValueError(f"Graph {task.task_graph} not found in {csv_file}.")
            task_graph = task_graphs[task.task_graph]
            is_task_good = task_graph.was_completed and not task_graph.missed_deadline
            for placement in task.placements:
                for resource in placement.resources_used:
                    if resource.name not in resource_utilization:
                        # The format is (good, bad) utilization.
                        resource_utilization[resource.name] = defaultdict(
                            lambda: [0, 0]
                        )

                    for t in range(placement.placement_time, placement.completion_time):
                        if is_task_good:
                            resource_utilization[resource.name][t][
                                0
                            ] += resource.quantity
                        else:
                            resource_utilization[resource.name][t][
                                1
                            ] += resource.quantity
                        if t > simulator_end_time:
                            simulator_end_time = t

        for resource in resource_utilization.keys():
            worker_pools = csv_reader.get_worker_pools(csv_file)
            max_resource_available = 0
            for worker_pool in worker_pools:
                for wp_resource in worker_pool.resources:
                    if resource == wp_resource.name:
                        max_resource_available += wp_resource.quantity
            usage_map = []
            print(f"Resource {resource} used between 0 and {simulator_end_time}.")
            total_good_utilization = 0
            total_utilization = 0
            for t in range(0, simulator_end_time):
                if t in resource_utilization[resource]:
                    good_utilization = resource_utilization[resource][t][0]
                    total_good_utilization += good_utilization
                    bad_utilization = resource_utilization[resource][t][1]
                    total_utilization += good_utilization + bad_utilization
                    usage_map.append((good_utilization, bad_utilization))
                else:
                    usage_map.append((0, 0))

            average_good_utilization = (
                total_good_utilization / (max_resource_available * simulator_end_time)
            ) * 100
            print(f"Average Good Utilization: {average_good_utilization}")
            average_total_utilization = (
                total_utilization / (max_resource_available * simulator_end_time)
            ) * 100
            print(f"Average Utilization: {average_total_utilization}")
            if plot:
                plt.figure(figsize=(8, 8))
                plt.bar(
                    np.arange(0, len(usage_map)),
                    [good_utilization for good_utilization, _ in usage_map],
                    color="green",
                    width=1.0,
                )
                plt.bar(
                    np.arange(0, len(usage_map)),
                    [bad_utilization for _, bad_utilization in usage_map],
                    bottom=[good_utilization for good_utilization, _ in usage_map],
                    color="red",
                    width=1.0,
                )
                plt.xlabel("Time")
                plt.ylabel("Resource Utilization")
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"{resource}_{scheduler_label}_{output_file_name}",
                    ),
                    bbox_inches="tight",
                )


def log_aggregate_stats(
    csv_reader, csv_files, conf_files, scheduler_labels, task_name_regex, stat="p50"
):
    """Prints the aggregate statistics for the given sequence of CSV files.

    Args:
        csv_reader (:py:class:`CSVReader`): The CSVReader instance containing the
            results.
        csv_files (Sequence[str]): The path to the CSV files to show the results for.
        conf_files (Sequence[str]): The path to the conf files to show the results for.
        scheduler_labels (Sequence[str]): The labels to assign to the schedulers.
        task_name_regex (`str`): The regular expression to match the task name to.
        stat (str): The stat to show in the table for the given tasks.
    """
    # Get the Statistic function to use.
    if stat not in STATS_FUNCTIONS:
        raise ValueError(
            f"Requested stat: {stat} not found in {STATS_FUNCTIONS.keys()}"
        )
    stat_function, stat_description = STATS_FUNCTIONS[stat]

    results = []
    for csv_file, conf_file, scheduler_label in zip(
        csv_files, conf_files, scheduler_labels
    ):
        __parse_flagfile(conf_file)
        log_name = Path(csv_file).stem
        tasks = []
        for task in csv_reader.get_tasks(csv_file):
            if re.match(task_name_regex, task.name):
                tasks.append(task)

        num_timestamps = FLAGS.max_timestamp if FLAGS.max_timestamp else "-"
        num_missed = len(list(filter(attrgetter("missed_deadline"), tasks)))
        num_skipped = len(list(filter(lambda task: len(task.skipped_times) > 0, tasks)))

        placement_delay = stat_function(
            [task.get_placement_delay() / 1000 for task in tasks]
        )
        deadline_delay = stat_function(
            [task.get_deadline_delay() / 1000 for task in tasks]
        )

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
        e2e_response_time = [
            (ct - rt) / 1000 for (rt, ct) in timestamp_start_end.values()
        ]

        worker_pool_stats = csv_reader.get_worker_pool_utilizations(csv_file)
        resource_uses = {
            resource: [
                stat.resource_utilizations[resource][0]
                / sum(stat.resource_utilizations[resource])
                for stat in worker_pool_stats
            ]
            for resource in ("Slot",)
        }

        scheduler_invocations = csv_reader.get_scheduler_invocations(csv_file)
        placed_tasks = [
            scheduler_invocation.num_placed_tasks
            for scheduler_invocation in scheduler_invocations
        ]
        unplaced_tasks = [
            scheduler_invocation.num_unplaced_tasks
            for scheduler_invocation in scheduler_invocations
        ]

        results.append(
            (
                num_timestamps,
                len(tasks),
                num_missed,
                num_skipped,
                placement_delay,
                deadline_delay,
                stat_function(e2e_response_time),
                stat_function(resource_uses["Slot"]),
                stat_function(placed_tasks),
                stat_function(unplaced_tasks),
                log_name,
            )
        )

    logger.debug(
        f"Aggregated results for tasks {task_name_regex} with {stat_description}: \n"
        + tabulate(
            list(sorted(results, key=lambda v: v[2])),
            headers=[
                "# Time",
                "# Tasks",
                "# Missed",
                "# Skipped",
                "Placement",
                "Deadline",
                "JCT",
                "Slot",
                "Placed",
                "Unplaced",
                "Log",
            ],
            tablefmt="grid",
            showindex=True,
        )
    )


def main(argv):
    # Parse the flags.
    try:
        re.compile(FLAGS.task_name)
    except re.error:
        raise ValueError(
            f"The regular expression for Task names: {FLAGS.task_name} is invalid."
        )

    statistics = (
        "all" if len(FLAGS.stat) == 1 and FLAGS.stat[0] == "all" else FLAGS.stat
    )

    # Get the labels from either the csv_labels or conf_files.
    if FLAGS.csv_labels:
        scheduler_labels = FLAGS.csv_labels
    else:
        scheduler_labels = []
        for i, (csv_file, conf_file) in enumerate(
            zip(FLAGS.csv_files, FLAGS.conf_files)
        ):
            __parse_flagfile(conf_file)
            scheduler_label = FLAGS.scheduler + f"_{i}"
            scheduler_labels.append(scheduler_label)

    figure_size = (14, 10)

    # Load the events from the CSV file into the CSVReader class.
    csv_reader = CSVReader(csv_paths=FLAGS.csv_files)

    for scheduler_csv_file, scheduler_label in zip(FLAGS.csv_files, scheduler_labels):
        # Log basic statistics
        simulation_end_time = csv_reader.get_simulator_end_time(scheduler_csv_file)
        logger.debug(
            f"Simulation end time for {scheduler_csv_file}: {simulation_end_time}"
        )

        # Log the basic statistics if requested.
        if FLAGS.task_stats == "basic":
            if len(statistics) != 1:
                _statistic = "p50"
            else:
                _statistic = statistics[0] if statistics[0] != "all" else "p50"
            log_basic_task_statistics(
                FLAGS.task_name, csv_reader, scheduler_csv_file, _statistic
            )

        # Log the detailed statistics if requested.
        if FLAGS.task_stats == "detailed" or FLAGS.all:
            log_detailed_task_statistics(
                FLAGS.task_name, csv_reader, scheduler_csv_file
            )

        # Output the Chrome trace format if requested.
        if FLAGS.chrome_trace:
            filename = Path(scheduler_csv_file).stem
            output_path = os.path.join(
                FLAGS.output_dir, filename + f"_{FLAGS.chrome_trace}.json"
            )
            logger.debug(f"Saving trace for {scheduler_csv_file} at {output_path}")
            if FLAGS.between_time and len(FLAGS.between_time) == 1:
                between_time = int(FLAGS.between_time[0])
            elif FLAGS.between_time:
                between_time = tuple(map(int, FLAGS.between_time))
            else:
                between_time = None
            csv_reader.to_chrome_trace(
                scheduler_csv_file,
                scheduler_label,
                output_path,
                between_time=between_time,
                trace_fmt=FLAGS.chrome_trace,
                show_release_times=(
                    FLAGS.show_release_times if FLAGS.show_release_times else "never"
                ),
                show_deadlines=(
                    FLAGS.show_deadlines if FLAGS.show_deadlines else "missed"
                ),
                with_placement_issues=FLAGS.with_placement_issues,
            )

        # Show statistics or plot the requested graphs.
        if FLAGS.resource_utilization or FLAGS.all:
            analyze_resource_utilization(
                csv_reader,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_"
                    f"{FLAGS.resource_utilization_timeline_plot_name}",
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )
        if FLAGS.task_placement or FLAGS.all:
            analyze_task_placement(
                csv_reader,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.task_placement_bar_chart_plot_name}",
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )
        if FLAGS.task_slack or FLAGS.all:
            analyze_task_slack(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir, f"{scheduler_label}_{FLAGS.task_slack_plot_name}"
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )
        if FLAGS.inter_task_time or FLAGS.all:
            analyze_inter_task_time(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.inter_task_time_plot_name}",
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )
        if FLAGS.missed_deadlines or FLAGS.all:
            analyze_missed_deadlines(
                csv_reader,
                FLAGS.task_name,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.missed_deadline_plot_name}",
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )
        if FLAGS.plot_goodput_graph or FLAGS.all:
            plot_goodput(
                csv_reader,
                FLAGS.csv_files,
                scheduler_labels,
                os.path.join(
                    FLAGS.output_dir,
                    f"{FLAGS.goodput_plot_name}",
                ),
                title=FLAGS.goodput_plot_title,
            )
        if FLAGS.end_to_end_response_time or FLAGS.all:
            analyze_end_to_end_response_time(
                csv_reader,
                scheduler_csv_file,
                scheduler_label,
                os.path.join(
                    FLAGS.output_dir,
                    f"{scheduler_label}_{FLAGS.end_to_end_response_time_plot_name}",
                ),
                plot=FLAGS.plot,
                figure_size=figure_size,
                stats=statistics,
            )

    if FLAGS.sched_runtime or FLAGS.all:
        analyze_scheduler_runtime(
            csv_reader,
            FLAGS.csv_files,
            scheduler_labels,
            os.path.join(FLAGS.output_dir, FLAGS.sched_runtime_timeline_plot_name),
            os.path.join(FLAGS.output_dir, FLAGS.sched_runtime_cdf_plot_name),
            plot=FLAGS.plot,
            figure_size=figure_size,
            stats=statistics,
        )
    if FLAGS.task_placement_delay or FLAGS.all:
        analyze_task_placement_delay(
            csv_reader,
            FLAGS.csv_files,
            scheduler_labels,
            os.path.join(FLAGS.output_dir, FLAGS.task_placement_delay_plot_name),
            plot=FLAGS.plot,
            figure_size=figure_size,
            stats=statistics,
        )
    if FLAGS.goodresource_utilization or FLAGS.all:
        plot_goodresource_utilization(
            csv_reader,
            FLAGS.csv_files,
            scheduler_labels,
            FLAGS.plot,
            FLAGS.output_dir,
            FLAGS.goodresource_utilization_plot_name,
        )

    if FLAGS.aggregate_stats:
        if len(statistics) != 1:
            _statistic = "p50"
        else:
            _statistic = statistics[0] if statistics[0] != "all" else "p50"
        log_aggregate_stats(
            csv_reader,
            FLAGS.csv_files,
            FLAGS.conf_files,
            scheduler_labels,
            FLAGS.task_name,
            _statistic,
        )


if __name__ == "__main__":
    app.run(main)
