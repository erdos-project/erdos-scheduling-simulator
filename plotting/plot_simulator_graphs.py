import sys
from collections import defaultdict
from operator import attrgetter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.patches import Patch

from data import Plotter
from utils import log_statistics, setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "csv_files", ["../test.csv"], "List of CSV files containing experiment logs"
)
flags.DEFINE_list(
    "csv_labels", ["test_scheduler"], "List of labels to use for the experiment logs"
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

matplotlib.rcParams.update({"font.size": 16, "figure.autolayout": True})
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.family"] = "serif"
axes_fontsize = 16
markers = {"EDF": "o", "Gurobi": "x", "Z3": "+", "LSF": "^"}
colors = {"EDF": "r", "Gurobi": "b", "Z3": "y", "LSF": "b"}

logger = setup_logging("plotting")


def plot_utilization(
    plotter,
    scheduler_csv_file,
    scheduler_name,
    output,
    figure_size=(14, 10),
    bar_width=1.0,
):
    # Plotting defaults.
    # hatches = ['//', '--', '**']
    # alphas = np.arange(0.2, 1.2, 0.2)
    resource_color = {"GPU": "red", "CPU": "green"}

    # Worker Pool statistics
    worker_pool_stats = plotter.get_worker_pool_utilizations(scheduler_csv_file)

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

    for resource_type in resource_types:
        logger.debug(
            "Max {} resource utilization {}".format(
                resource_type, max(resource_used_heights[resource_type])
            )
        )

    # Plot a histogram with the results.
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


def plot_scheduler_runtime(plotter, figure_size=(14, 10)):
    # Retrieve the runtime of the scheduler invocations.
    logger.debug("================= Scheduler runtime [ms] =================")
    max_start_time = -sys.maxsize
    max_runtime = -sys.maxsize
    all_runtimes = []
    all_start_times = []
    for csv_file in FLAGS.csv_files:
        scheduler_invocations = plotter.get_scheduler_invocations(csv_file)
        runtimes = list(map(attrgetter("runtime"), scheduler_invocations))
        runtimes_ms = [runtime / 1000 for runtime in runtimes]
        start_times = list(map(attrgetter("start_time"), scheduler_invocations))
        start_times_ms = [start_time / 1000 for start_time in start_times]
        max_start_time = max(max_start_time, max(start_times_ms))
        max_runtime = max(max_runtime, max(runtimes_ms))
        all_runtimes.append(runtimes_ms)
        all_start_times.append(start_times_ms)
        logger.debug(f"Scheduler runtime stats {csv_file}")
        log_statistics(runtimes_ms, logger)

    # Plot a timelapse of the runtime of the scheduler.
    plt.figure(figsize=figure_size)
    for i, label in enumerate(FLAGS.csv_labels):
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
    plt.savefig(FLAGS.scheduler_runtime_timeline_plot_name, bbox_inches="tight")

    # Plot the CDF of the runtime of the scheduler invocations.

    plt.figure(figsize=figure_size)
    for i, label in enumerate(FLAGS.csv_labels):
        count, bin_count = np.histogram(all_runtimes[i], bins=100)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        plt.plot(bin_count[1:], cdf, label=label)
    plt.xlim(0, max_runtime)
    plt.xlabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
    plt.ylabel("CDF", fontsize=axes_fontsize)
    plt.legend(frameon=False)
    plt.savefig(FLAGS.scheduler_runtime_cdf_plot_name, bbox_inches="tight")


def plot_task_placement_stats(
    plotter, scheduler_csv_file, scheduler_name, output, figure_size=(14, 10)
):
    scheduler_invocations = plotter.get_scheduler_invocations(scheduler_csv_file)
    # Calculate the heights of placed and unplaced tasks.
    placed_task_heights = [
        scheduler_invocation.placed_tasks
        for scheduler_invocation in scheduler_invocations
    ]
    unplaced_task_heights = [
        scheduler_invocation.unplaced_tasks
        for scheduler_invocation in scheduler_invocations
    ]

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

    logger.debug("================== Num placed tasks ==================")
    log_statistics(placed_task_heights, logger)
    logger.debug("================== Num unplaced tasks ==================")
    log_statistics(unplaced_task_heights, logger)

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
    plotter, scheduler_csv_file, scheduler_name, output, figure_size=(14, 10)
):
    plt.figure(figsize=figure_size)
    tasks = plotter.get_tasks(scheduler_csv_file)
    task_map = defaultdict(list)
    for task in tasks:
        task_map[task.name].append(task)

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

    plt.xlabel("Inter-Task Time [ms]", fontsize=axes_fontsize)
    plt.hist(inter_release_times, label=labels, density=False, bins=100)
    plt.legend(frameon=False)
    plt.savefig(output, bbox_inches="tight")


def plot_task_slack(plotter, csv_file, scheduler_name, output, figure_size=(14, 10)):
    # Plot a histogram of the slack from the deadline for the tasks.
    plt.figure(figsize=figure_size)
    tasks = plotter.get_tasks(csv_file)
    slack = [(task.deadline - task.completion_time) / 1000 for task in tasks]
    initial_slack = [
        (task.deadline - task.release_time - task.runtime) / 1000 for task in tasks
    ]
    logger.debug("================== Task completion slack [ms] ==================")
    log_statistics(slack, logger)
    logger.debug("================== Task placement slack [ms] ==================")
    log_statistics(initial_slack, logger)
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


def plot_task_placement_delay(plotter, figure_size=(14, 10)):
    plt.figure(figsize=figure_size)
    logger.debug("================ Task placement delay [ms] ================")
    placement_delays = []
    min_delay = sys.maxsize
    max_delay = -sys.maxsize
    plot_colors = [colors[label] for label in FLAGS.csv_labels]
    for csv_file in FLAGS.csv_files:
        task_placements = plotter.get_task_placements(csv_file)
        placement_delay = [
            (placement.simulator_time - placement.task.release_time) / 1000
            for placement in task_placements
        ]
        placement_delays.append(placement_delay)
        min_delay = min(min_delay, min(placement_delay))
        max_delay = max(max_delay, max(placement_delay))
        logger.debug(f"Placement delay stats {csv_file}")
        log_statistics(placement_delay, logger)

    plt.xlim(min_delay, max_delay)
    plt.xlabel("Task Placement Delay [ms]", fontsize=axes_fontsize)
    plt.ylabel("Number of Tasks", fontsize=axes_fontsize)
    plt.hist(
        placement_delays,
        density=False,
        bins=100,
        label=FLAGS.csv_labels,
        color=plot_colors,
    )
    plt.legend(frameon=False)
    plt.savefig(FLAGS.task_placement_delay_plot_name, bbox_inches="tight")


def task_stats(tasks):
    for task in tasks:
        logger.debug(f"Task Name: {task.name}\n\tRelease Time: {task.release_time}\n\t")
        logger.debug(
            f"Completion Time: {task.completion_time}\n\tDeadline: {task.deadline}\n\t"
        )
        logger.debug(
            f"Runtime: {task.runtime}\n\tSlack: {task.deadline - task.completion_time}"
        )


def main(argv):
    assert len(FLAGS.csv_files) == len(
        FLAGS.csv_labels
    ), "Mismatch between length of csv files and labels flags."
    figure_size = (14, 10)

    # Load the events from the CSV file into the Plotter class.
    plotter = Plotter(csv_paths=FLAGS.csv_files)

    for csv_file in FLAGS.csv_files:
        simulation_end_time = plotter.get_simulator_end_time(csv_file)
        logger.debug(f"Simulation end time for {csv_file} : {simulation_end_time}")
        logger.debug(
            f"Number of tasks for {csv_file}: {len(plotter.get_tasks(csv_file))}"
        )
        # Missed deadline statistics
        missed_deadline_events = plotter.get_missed_deadline_events(csv_file)
        logger.debug(
            f"Number of missed deadlines for {csv_file}: {len(missed_deadline_events)}"
        )

    for scheduler_csv_file, scheduler_label in zip(FLAGS.csv_files, FLAGS.csv_labels):
        if FLAGS.plot_utilization:
            plot_utilization(
                plotter,
                scheduler_csv_file,
                scheduler_label,
                f"{scheduler_label}_{FLAGS.utilization_timeline_plot_name}",
                figure_size,
            )
        if FLAGS.plot_task_placement_stats:
            plot_task_placement_stats(
                plotter,
                scheduler_csv_file,
                scheduler_label,
                f"{scheduler_label}_{FLAGS.task_placement_bar_chart_plot_name}",
                figure_size,
            )
        if FLAGS.plot_task_slack:
            plot_task_slack(
                plotter,
                scheduler_csv_file,
                scheduler_label,
                f"{scheduler_label}_{FLAGS.task_slack_plot_name}",
                figure_size,
            )
        if FLAGS.plot_inter_task_time:
            plot_inter_task_time(
                plotter,
                scheduler_csv_file,
                scheduler_label,
                f"{scheduler_label}_{FLAGS.inter_task_time_plot_name}",
                figure_size,
            )

    if FLAGS.plot_scheduler_runtime:
        plot_scheduler_runtime(plotter, figure_size)
    if FLAGS.plot_task_placement_delay:
        plot_task_placement_delay(plotter, figure_size)


if __name__ == "__main__":
    app.run(main)
