from operator import attrgetter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.patches import Patch

from data import Plotter
from utils import log_statistics, setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("csv_file", "../test.csv", "CSV file containing experiment logs")
flags.DEFINE_bool("plot_scheduler_runtime", False, "Plot scheduling runtime")
flags.DEFINE_bool("plot_utilization", False, "Plot resource utilization")
flags.DEFINE_bool("plot_task_info", False, "Plot task placement info")
flags.DEFINE_bool("plot_task_slack", False, "Plot task slack")
flags.DEFINE_bool("plot_task_placement_delay", False, "Plot task placement delay")
flags.DEFINE_string(
    "utilization_timeline_plot_name",
    "utilization_bar_chart.png",
    "The filename of the utilization timeline plot",
)
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
flags.DEFINE_string(
    "task_placement_bar_chart_plot_name",
    "tasks_placed_bar_chart.png",
    "The filename of the task placement bar char plot",
)
flags.DEFINE_string(
    "task_slack_plot_name", "task_slack.png", "The filename of the task slack plot"
)
flags.DEFINE_string(
    "task_placement_delay_plot_name",
    "task_placement_delay.png",
    "The filename of the task start delay plot",
)

matplotlib.rcParams.update({"font.size": 16, "figure.autolayout": True})
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.family"] = "serif"
axes_fontsize = 16

logger = setup_logging("plotting")


def plot_utilization(plotter, figure_size=(14, 10), bar_width=1.0):
    # Plotting defaults.
    # hatches = ['//', '--', '**']
    # alphas = np.arange(0.2, 1.2, 0.2)
    resource_color = {"GPU": "red", "CPU": "green"}

    # Worker Pool statistics
    worker_pool_stats = plotter.get_worker_pool_utilizations(FLAGS.csv_file)

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
    x_vals = np.arange(1, len(worker_pool_stats) + 1)
    _ = [
        plt.bar(
            x_vals + bar_width * i,
            resource_used_heights[resource_type],
            width=bar_width,
            # edgecolor='black',
            # hatch=hatches[i],
            alpha=0.4,
            color=resource_color[resource_type],
        )
        for i, resource_type in enumerate(resource_types)
    ]

    plt.xlabel("Scheduler run number", fontsize=axes_fontsize)
    plt.ylabel("Normalized utilization", fontsize=axes_fontsize)
    # Set the y limits so we can visualize the legend.
    ax = fig.gca()
    ax.set_ylim(0, 1.01)
    plt.yticks([0, 0.5, 1.0])

    legend_elements = [
        Patch(facecolor=resource_color[resource_type], label="{}".format(resource_type))
        for resource_type in resource_types
    ]
    plt.legend(handles=legend_elements, prop={"size": 28}, framealpha=0)
    plt.savefig(FLAGS.utilization_timeline_plot_name, bbox_inches="tight")


def plot_scheduler_runtime(plotter, figure_size=(14, 10)):
    # Retrieve the runtime of the scheduler invocations.
    scheduler_invocations = plotter.get_scheduler_invocations(FLAGS.csv_file)
    runtimes = list(map(attrgetter("runtime"), scheduler_invocations))
    runtimes_ms = [runtime / 1000 for runtime in runtimes]
    start_times = list(map(attrgetter("start_time"), scheduler_invocations))
    start_times_ms = [start_time / 1000 for start_time in start_times]
    logger.debug("================= Scheduler runtime [ms] =================")
    log_statistics(runtimes_ms, logger)
    # Plot a timelapse of the runtime of the scheduler.
    plt.figure(figsize=figure_size)
    plt.plot(start_times_ms, runtimes_ms, marker="o")
    plt.xlim(0, int(max(start_times_ms)))
    plt.xlabel("Timeline [ms]", fontsize=axes_fontsize)
    plt.ylabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
    plt.savefig(FLAGS.scheduler_runtime_timeline_plot_name, bbox_inches="tight")

    # Plot the CDF of the runtime of the scheduler invocations.
    count, bin_count = np.histogram(runtimes_ms, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    plt.figure(figsize=figure_size)
    plt.plot(bin_count[1:], cdf, label="CDF")
    plt.xlim(0, max(runtimes))
    plt.xlabel("Scheduler Runtime [ms]", fontsize=axes_fontsize)
    plt.ylabel("CDF", fontsize=axes_fontsize)
    plt.legend(prop={"size": 28})
    plt.savefig(FLAGS.scheduler_runtime_cdf_plot_name, bbox_inches="tight")


def plot_task_info(plotter, figure_size=(14, 10)):
    scheduler_invocations = plotter.get_scheduler_invocations(FLAGS.csv_file)
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
    y_ticks = list(range(int(max_y_val)))
    plt.yticks(y_ticks)

    # Set the legend.
    legend_elements = [
        Patch(facecolor="green", hatch="//", label="Placed Tasks"),
        Patch(facecolor="red", hatch="**", label="Unplaced Tasks"),
    ]
    plt.legend(handles=legend_elements, prop={"size": 28}, framealpha=0)
    plt.savefig(FLAGS.task_placement_bar_chart_plot_name, bbox_inches="tight")


def plot_task_slack(plotter, figure_size=(14, 10)):
    # Plot a histogram of the slack from the deadline for the tasks.
    plt.figure(figsize=figure_size)
    tasks = plotter.get_tasks(FLAGS.csv_file)
    slack = [(task.deadline - task.completion_time) / 1000 for task in tasks]
    logger.debug("================== Task slack [ms] ==================")
    log_statistics(slack, logger)
    plt.xlim(min(slack), max(slack))
    plt.xlabel("Task Slack [ms]", fontsize=axes_fontsize)
    plt.ylabel("Relative Frequency", fontsize=axes_fontsize)
    plt.hist(slack, density=True, bins=100)
    plt.savefig(FLAGS.task_slack_plot_name, bbox_inches="tight")


def plot_task_placement_delay(plotter, figure_size=(14, 10)):
    plt.figure(figsize=figure_size)
    task_placements = plotter.get_task_placements(FLAGS.csv_file)
    placement_delay = [
        (placement.simulator_time - placement.task.release_time) / 1000
        for placement in task_placements
    ]
    logger.debug("================ Task placement delay [ms] ================")
    log_statistics(placement_delay, logger)
    plt.xlim(min(placement_delay), max(placement_delay))
    plt.xlabel("Task Placement Delay [ms]", fontsize=axes_fontsize)
    plt.ylabel("Relative Frequency", fontsize=axes_fontsize)
    plt.hist(placement_delay, density=True, bins=100)
    plt.savefig(FLAGS.task_placement_delay_plot_name, bbox_inches="tight")


def task_stats(tasks):
    for task in tasks:
        logger.debug(
            "Task Name: {}\n\tRelease Time: {}\n\t".format(
                task.task_name, task.release_time
            )
        )
        logger.debug(
            "Completion Time: {}\n\tDeadline: {}\n\t".format(
                task.completion_time, task.deadline
            )
        )
        logger.debug(
            "Runtime: {}\n\tSlack: {}".format(
                task.runtime, task.deadline - task.completion_time
            )
        )


def main(argv):
    figure_size = (14, 10)
    # Load the events from the CSV file into the Plotter class.
    plotter = Plotter(csv_paths=[FLAGS.csv_file])
    # events = plotter._events[FLAGS.csv_file]

    logger.debug(
        "Simulation end time: {}".format(plotter.get_simulator_end_time(FLAGS.csv_file))
    )
    logger.debug("Number of tasks: {}".format(len(plotter.get_tasks(FLAGS.csv_file))))

    # Missed deadline statistics
    missed_deadline_events = plotter.get_missed_deadline_events(FLAGS.csv_file)
    logger.debug("Number of missed deadlines: {}".format(len(missed_deadline_events)))

    if FLAGS.plot_utilization:
        plot_utilization(plotter, figure_size)
    if FLAGS.plot_scheduler_runtime:
        plot_scheduler_runtime(plotter, figure_size)
    if FLAGS.plot_task_info:
        plot_task_info(plotter, figure_size)
    if FLAGS.plot_task_slack:
        plot_task_slack(plotter, figure_size)
    if FLAGS.plot_task_placement_delay:
        plot_task_placement_delay(plotter, figure_size)


if __name__ == "__main__":
    app.run(main)