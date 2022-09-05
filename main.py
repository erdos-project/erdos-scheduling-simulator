import random
import sys

from absl import app, flags

from data import (
    TaskLoaderBenchmark,
    TaskLoaderPylot,
    TaskLoaderSynthetic,
    WorkerLoaderBenchmark,
    WorkerLoaderJSON,
    WorkloadLoader,
)
from schedulers import (
    BranchPredictionScheduler,
    EDFScheduler,
    GurobiScheduler2,
    LSFScheduler,
    Z3Scheduler,
)
from simulator import Simulator
from utils import EventTime, setup_logging
from workload.workload import Workload

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_enum(
    "execution_mode",
    "replay",
    ["replay", "synthetic", "benchmark", "json"],
    "Sets the execution mode of the simulator. In the replay mode the simulator "
    "replays a Pylot log, in the synthetic mode the simulator generates a synthetic "
    "Pylot-like task workload, and in the benchmark mode the simulator generates a "
    "synthetic task workload. 'json' reads an abstract workload definition from a "
    "JSON file and simulates its execution.",
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
    "workload_profile_path",
    "./profiles/workload/pylot-workload-profile.json",
    "Path of the description of the Workload to schedule.",
)
flags.DEFINE_string(
    "worker_profile_path",
    "./profiles/workers/worker_profile.json",
    "Path of the topology of Workers to schedule on.",
)
flags.DEFINE_string(
    "profile_path",
    "./profiles/workload/pylot_profile.json",
    "Path of the JSON profile for the Pylot execution.",
)
flags.DEFINE_bool("stats", False, "Print the statistics from the tasks loaded.")

# Simulator related flags.
flags.DEFINE_integer(
    "loop_timeout", sys.maxsize, "Timeout for the Simulator loop (in us)."
)
flags.DEFINE_integer(
    "random_seed", 42, "The seed to be used for random number generation."
)

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
flags.DEFINE_bool(
    "use_branch_predicated_deadlines",
    False,
    "If True, then the end-to-end deadline is assigned based on the branches expected "
    "to execute.",
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
    ["EDF", "LSF", "Gurobi", "Z3", "BranchPrediction"],
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
flags.DEFINE_enum(
    "scheduler_policy",
    "worst",
    ["best", "worst", "random"],
    "The policy to be used for the BranchPredictionScheduler.",
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
    random.seed(FLAGS.random_seed)
    logger = setup_logging(
        name=__name__, log_file=FLAGS.log_file_name, log_level=FLAGS.log_level
    )
    logger.info("Starting the execution of the simulator loop.")
    logger.info("Workload File: %s", FLAGS.workload_profile_path)
    logger.info("Workers File: %s", FLAGS.worker_profile_path)
    logger.info("Profile File: %s", FLAGS.profile_path)

    # Load the data.
    if FLAGS.execution_mode == "replay":
        workload_loader = WorkloadLoader(
            json_path=FLAGS.workload_profile_path, _flags=FLAGS
        )
        job_graph = workload_loader.workload.get_job_graph("pylot_dataflow")
        task_loader = TaskLoaderPylot(
            job_graph=job_graph,
            graph_name="pylot_dataflow",
            profile_path=FLAGS.profile_path,
            _flags=FLAGS,
        )
        workload = Workload.from_task_graphs(
            {"pylot_dataflow": task_loader.get_task_graph()},
            _flags=FLAGS,
        )
    elif FLAGS.execution_mode == "synthetic":
        task_loader = TaskLoaderSynthetic(
            num_perception_sensors=2,
            num_traffic_light_cameras=1,
            _flags=FLAGS,
        )
        raise NotImplementedError("Workload has not been specified yet.")
    elif FLAGS.execution_mode == "benchmark":
        task_loader = TaskLoaderBenchmark(
            num_jobs=5,
            task_runtime=FLAGS.benchmark_task_runtime,
            task_deadline=FLAGS.benchmark_task_deadline,
            _flags=FLAGS,
        )
        raise NotImplementedError("Workload has not been specified yet.")
    elif FLAGS.execution_mode == "json":
        workload_loader = WorkloadLoader(
            json_path=FLAGS.workload_profile_path, _flags=FLAGS
        )
        workload = workload_loader.workload

    # Dilate the time if needed.
    if FLAGS.timestamp_difference != -1:
        task_loader.get_task_graph().dilate(
            EventTime(FLAGS.timestamp_difference, EventTime.Unit.US)
        )

    if FLAGS.stats:
        if FLAGS.execution_mode != "json":
            # Only JSON execution mode generates workloads according to the given Job
            # workload, and does not have a pre-determined JobGraph.
            task_loader.log_statistics()
        else:
            # TODO (Sukrit): We should be implementing a statistics method for the
            # Workload too.
            pass
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
    elif FLAGS.scheduler == "BranchPrediction":
        if FLAGS.scheduler_policy == "best":
            policy = BranchPredictionScheduler.Policy.BEST_CASE
        elif FLAGS.scheduler_policy == "worst":
            policy = BranchPredictionScheduler.Policy.WORST_CASE
        elif FLAGS.scheduler_policy == "random":
            policy = BranchPredictionScheduler.Policy.RANDOM
        else:
            raise NotImplementedError(
                f"The policy {FLAGS.scheduler_policy} is not supported."
            )
        print(policy)

        scheduler = BranchPredictionScheduler(
            policy=policy,
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            _flags=FLAGS,
        )
    else:
        raise ValueError(
            "Unsupported scheduler implementation: {}".format(FLAGS.scheduler)
        )

    # Load the worker topology.
    if FLAGS.execution_mode in ["replay", "synthetic", "json"]:
        worker_loader = WorkerLoaderJSON(
            worker_profile_path=FLAGS.worker_profile_path, _flags=FLAGS
        )
    elif FLAGS.execution_mode == "benchmark":
        worker_loader = WorkerLoaderBenchmark(
            scheduler, FLAGS.benchmark_num_cpus, FLAGS.benchmark_num_gpus, _flags=FLAGS
        )
    else:
        raise NotImplementedError(
            f"WorkerLoader for execution mode {FLAGS.execution_mode} "
            "was not implemented."
        )

    # Create and run the Simulator based on the scheduler.
    simulator = Simulator(
        worker_pools=worker_loader.get_worker_pools(),
        scheduler=scheduler,
        workload=workload,
        loop_timeout=EventTime(FLAGS.loop_timeout, EventTime.Unit.US),
        _flags=FLAGS,
    )
    simulator.simulate()


if __name__ == "__main__":
    app.run(main)
