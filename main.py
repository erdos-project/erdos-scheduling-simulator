import random
import sys

from absl import app, flags

from data import (
    TaskLoaderBenchmark,
    TaskLoaderPylot,
    TaskLoaderSynthetic,
    WorkerLoaderBenchmark,
    WorkerLoaderJSON,
)
from schedulers import EDFScheduler, GurobiScheduler2, LSFScheduler, Z3Scheduler
from simulator import Simulator
from utils import EventTime, setup_logging

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_enum(
    "execution_mode",
    "replay",
    ["replay", "synthetic", "benchmark"],
    "Sets the execution mode of the simulator. In the replay mode the simulator "
    "replays a Pylot log, in the synthetic mode the simulator generates a synthetic "
    "Pylot-like task workload, and in the benchmark mode the simulator generates a "
    "synthetic task workload.",
)
flags.DEFINE_string(
    "log_file_name", None, "Name of the file to log the results to.", short_name="log"
)
flags.DEFINE_string(
    "csv_file_name",
    None,
    "Name of the CSV file to log the results to.",
    short_name="csv",
)
flags.DEFINE_string(
    "graph_file_name", None, "Name of the Graph file to log the JobGraph to."
)
flags.DEFINE_string(
    "scheduler_log_base_name",
    None,
    "Base name to use to log info about scheduler runs.",
)
flags.DEFINE_string("log_level", "debug", "Level of logging.")
flags.DEFINE_string(
    "graph_path",
    "./profiles/workload/pylot-complete-graph.dot",
    "Path of the DOT file that contains the execution graph.",
)
flags.DEFINE_string(
    "profile_path",
    "./profiles/workload/pylot_profile.json",
    "Path of the JSON profile for the Pylot execution.",
)
flags.DEFINE_string(
    "resource_path",
    "./profiles/workload/pylot_resource_profile.json",
    "Path of the Resource requirements for each Task.",
)
flags.DEFINE_string(
    "worker_profile_path",
    "./profiles/workers/worker_profile.json",
    "Path of the topology of Workers to schedule on.",
)
flags.DEFINE_bool("stats", False, "Print the statistics from the tasks loaded.")

# Benchmark related flags.
flags.DEFINE_integer(
    "benchmark_task_runtime", 15000, "Estimated runtime of benchmark tasks (in us)."
)
flags.DEFINE_integer(
    "benchmark_task_deadline", 500000, "Deadline of benchmark tasks (in us)."
)
flags.DEFINE_integer(
    "benchmark_num_gpus", 2, "Number of GPUs available for benchmarking."
)
flags.DEFINE_integer(
    "benchmark_num_cpus", 10, "Number of CPUs available for benchmarking."
)

# Task related flags.
flags.DEFINE_integer(
    "max_timestamp",
    None,
    "Maximum timestamp of tasks to load from the JSON file.",
    short_name="max",
)
flags.DEFINE_bool(
    "use_end_to_end_deadlines",
    False,
    "If True, then the end-to-end deadline for a JobGraph is assigned to tasks.",
)
flags.DEFINE_integer(
    "min_deadline_variance",
    0,
    "The minimum % variance to allocate to the assigned deadline for each task.",
)
flags.DEFINE_integer(
    "max_deadline_variance",
    20,
    "The maximum % variance to allocate to the assigned deadline for each task.",
)
flags.DEFINE_integer(
    "runtime_variance",
    0,
    "The % variance to allocate to the assigned runtime for each task.",
)
flags.DEFINE_integer(
    "timestamp_difference",
    -1,
    "The difference to keep between the source Jobs of successive timestamps.",
)
flags.DEFINE_bool(
    "synchronize_sensors", False, "If True then the sensor operators are synchronized."
)

# Scheduler related flags.
flags.DEFINE_enum(
    "scheduler",
    "EDF",
    ["EDF", "LSF", "Gurobi", "Z3"],
    "The scheduler to use for this execution.",
)
flags.DEFINE_bool(
    "preemption", False, "Enable preemption of running tasks in the scheduler."
)
flags.DEFINE_integer(
    "scheduler_runtime",
    -1,
    "The runtime (in us) to assign to each scheduler invocation.",
)
flags.DEFINE_integer(
    "scheduler_delay",
    1,
    "The delay (in us) associated with invoking a scheduler after the "
    "release of a Task in the system.",
)
flags.DEFINE_integer(
    "scheduler_lookahead",
    0,
    "The scheduler places tasks that are estimated to be released "
    "within the scheduling lookahead (in us).",
)
flags.DEFINE_bool(
    "enforce_deadlines",
    False,
    "True if the ILP formulation must ensure that deadlines are met.",
)
flags.DEFINE_enum(
    "ilp_goal",
    "max_slack",
    ["max_slack", "min_placement_delay"],
    "Sets the goal of the ILP solver.",
)


def main(args):
    """Main loop that loads the data from the given profile paths, and
    runs the Simulator on the data with the given scheduler.
    """
    random.seed(42)
    logger = setup_logging(
        name=__name__, log_file=FLAGS.log_file_name, log_level=FLAGS.log_level
    )
    logger.info("Starting the execution of the simulator loop.")
    logger.info("Graph File: %s", FLAGS.graph_path)
    logger.info("Profile File: %s", FLAGS.profile_path)
    logger.info("Resource File: %s", FLAGS.resource_path)

    # Load the data.
    if FLAGS.execution_mode == "replay":
        task_loader = TaskLoaderPylot(
            graph_path=FLAGS.graph_path,
            profile_path=FLAGS.profile_path,
            resource_path=FLAGS.resource_path,
            _flags=FLAGS,
        )
    elif FLAGS.execution_mode == "synthetic":
        task_loader = TaskLoaderSynthetic(
            num_perception_sensors=2,
            num_traffic_light_cameras=1,
            _flags=FLAGS,
        )
    elif FLAGS.execution_mode == "benchmark":
        task_loader = TaskLoaderBenchmark(
            num_jobs=5,
            task_runtime=FLAGS.benchmark_task_runtime,
            task_deadline=FLAGS.benchmark_task_deadline,
            _flags=FLAGS,
        )

    # Dilate the time if needed.
    if FLAGS.timestamp_difference != -1:
        task_loader.get_task_graph().dilate(
            EventTime(FLAGS.timestamp_difference, EventTime.Unit.US)
        )

    if FLAGS.stats:
        # Log the statistics, and do not execute the Simulator.
        task_loader.log_statistics()
        return
    if FLAGS.graph_file_name:
        # Log the graph to the given file and do not execute the Simulator.
        job_graph = task_loader.get_job_graph()
        job_graph.to_dot(FLAGS.graph_file_name)
        return

    # Instantiate the scheduler based on the given flag.
    scheduler = None
    if FLAGS.scheduler == "EDF":
        scheduler = EDFScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "LSF":
        scheduler = LSFScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "Gurobi":
        scheduler = GurobiScheduler2(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            goal=FLAGS.ilp_goal,
            enforce_deadlines=FLAGS.enforce_deadlines,
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            _flags=FLAGS,
            _time_unit=EventTime.Unit.MS,
        )
    elif FLAGS.scheduler == "Z3":
        scheduler = Z3Scheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            goal=FLAGS.ilp_goal,
            enforce_deadlines=FLAGS.enforce_deadlines,
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            _flags=FLAGS,
        )
    else:
        raise ValueError(
            "Unsupported scheduler implementation: {}".format(FLAGS.scheduler)
        )

    # Load the worker topology.
    if FLAGS.execution_mode in ["replay", "synthetic"]:
        worker_loader = WorkerLoaderJSON(
            worker_profile_path=FLAGS.worker_profile_path, _flags=FLAGS
        )
    elif FLAGS.execution_mode == "benchmark":
        worker_loader = WorkerLoaderBenchmark(
            scheduler, FLAGS.benchmark_num_cpus, FLAGS.benchmark_num_gpus, _flags=FLAGS
        )

    # Create and run the Simulator based on the scheduler.
    simulator = Simulator(
        worker_pools=worker_loader.get_worker_pools(),
        scheduler=scheduler,
        job_graph=task_loader.get_job_graph(),
        _flags=FLAGS,
    )
    simulator.simulate(task_graph=task_loader.get_task_graph())


if __name__ == "__main__":
    app.run(main)
