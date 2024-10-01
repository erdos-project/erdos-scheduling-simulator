import os
import random
import sys

from absl import app, flags

from data import (
    AlibabaLoader,
    TaskLoaderBenchmark,
    TaskLoaderPylot,
    TaskLoaderSynthetic,
    TpchLoader,
    WorkerLoader,
    WorkerLoaderBenchmark,
    WorkloadLoader,
    WorkloadLoaderClockworkBursty,
)
from simulator import Simulator
from utils import EventTime, setup_csv_logging, setup_logging
from workload import BranchPredictionPolicy, JobGraph, Workload

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_enum(
    "execution_mode",
    "replay",
    ["replay", "synthetic", "benchmark", "json", "yaml"],
    "Sets the execution mode of the simulator. In the replay mode the simulator "
    "replays a Pylot log, in the synthetic mode the simulator generates a synthetic "
    "Pylot-like task workload, and in the benchmark mode the simulator generates a "
    "synthetic task workload. 'json' / 'yaml' reads an abstract workload definition "
    "from a JSON / YAML file and simulates its execution.",
)
flags.DEFINE_enum(
    "replay_trace",
    "pylot",
    ["pylot", "clockwork_bursty", "alibaba", "tpch"],
    "Sets the trace to replay in the replay mode.",
)
flags.DEFINE_string(
    "log_file_name", None, "Name of the file to log the results to.", short_name="log"
)
flags.DEFINE_enum(
    "log_file_mode",
    "write",
    ["append", "write"],
    "Sets the mode in which the log file is opened. If 'append', the log file is "
    "opened in append mode, and if 'write', the log file is opened in write mode. ",
)
flags.DEFINE_string(
    "csv_file_name",
    None,
    "Name of the CSV file to log the results to.",
    short_name="csv",
)
flags.DEFINE_bool(
    "log_graphs",
    False,
    "If True, the simulator logs the TaskGraphs.",
)
flags.DEFINE_string("log_level", "debug", "Level of logging.")
flags.DEFINE_string(
    "workload_profile_path",
    None,
    "Path of the description of the Workload to schedule. The path can be a file, or a "
    "directory, and the behavior is defined by the workload loader used. Use "
    "`workload_profile_paths` if specific files are to be loaded.",
)
flags.DEFINE_list(
    "workload_profile_paths",
    [],
    "A list of paths of the description of the Workload to schedule.",
)
flags.DEFINE_list(
    "workload_profile_path_labels",
    [],
    "A list of labels for the workload profile paths. If provided, the list must be of "
    "the same length as the list of workload profile paths. This is used to annotate "
    "the TaskGraphs with the corresponding workload profile path label.",
)
flags.DEFINE_string(
    "worker_profile_path",
    "./profiles/workers/worker_profile.json",
    "Path of the topology of Workers to schedule on.",
)
flags.DEFINE_bool("stats", False, "Print the statistics from the tasks loaded.")
flags.DEFINE_bool("dry_run", False, "If True, the simulator does not run.")

# Simulator related flags.
flags.DEFINE_integer(
    "loop_timeout", sys.maxsize, "Timeout for the Simulator loop (in us)."
)
flags.DEFINE_integer(
    "random_seed",
    random.randint(0, sys.maxsize),
    "The seed to be used for random number generation. Defaults to a random number.",
)
flags.DEFINE_bool(
    "resolve_conditionals_at_submission",
    False,
    "True if the conditionals are to be resolved at the Job submission time "
    "instead of when the previous node finishes execution.",
)
flags.DEFINE_bool(
    "drop_skipped_tasks",
    False,
    "True if the simulator is allowed to drop tasks that cannot be scheduled."
    "This option can be used with SAT/ILP based schedulers to stop wastefully"
    "reconsidering tasks that will never meet their deadlines.",
)
flags.DEFINE_integer(
    "workload_update_interval",
    -1,
    "The interval (in µs) to update the workload. "
    "If set to default (-1), then the Simulator will automatically choose an interval "
    "based on the set of released tasks in the previous iteration.",
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

# TPCH related flags
flags.DEFINE_string(
    "tpch_query_dag_spec",
    "./profiles/workload/tpch/queries.yaml",
    "Path to a YAML file specifying the TPC-H query DAGs",
)
flags.DEFINE_integer(
    "tpch_num_queries",
    50,
    "Number of TPC-H queries to run",
)
flags.DEFINE_enum(
    "tpch_dataset_size",
    "50",
    ["2", "50", "100", "250", "500"],
    "Size of the TPC-H dataset to use",
)
flags.DEFINE_integer(
    "tpch_max_executors_per_job",
    50,
    "Maximum number of executors to use per TPC-H query stage",
)
flags.DEFINE_integer(
    "tpch_min_task_runtime",
    8,
    "Minimum runtime of a TPC-H task",
)

# AlibabaLoader related flags.
flags.DEFINE_integer(
    "alibaba_loader_task_cpu_multiplier",
    1,
    "The multiplier used on the resources after all the minimum and maximum clipping "
    "has been done.",
)
flags.DEFINE_integer(
    "alibaba_loader_task_cpu_divisor",
    25,
    "The divisor used when converting alibaba trace tasks task.cpu to slot. The lower "
    "the number, the higher the slots.",
)
flags.DEFINE_bool(
    "alibaba_bump_resources_of_low_duration_task",
    False,
    "If True, the lower duration task resources would be bumped to max",
)
flags.DEFINE_bool(
    "alibaba_loader_task_cpu_usage_random",
    False,
    "If True, the task CPU usage is randomly generated between the min and max values.",
)
flags.DEFINE_integer(
    "alibaba_loader_task_cpu_usage_min",
    0,
    "The minimum CPU usage of a task that can be released by the Alibaba trace. "
    "This property is verified AFTER the task CPU divisor is applied.",
)
flags.DEFINE_integer(
    "alibaba_loader_task_cpu_usage_max",
    sys.maxsize,
    "The maximum CPU usage of a task that can be released by the Alibaba trace. "
    "This property is verified AFTER the task CPU divisor is applied.",
)
flags.DEFINE_float(
    "alibaba_task_duration_multiplier",
    1,
    "The multiplier used for alibaba trace tasks task.duration.",
)
flags.DEFINE_list(
    "alibaba_loader_min_critical_path_runtimes",
    [],
    "The minimum critical path duration for each TaskGraph from the corresponding "
    "Workload. If the list is empty, then the minimum critical path duration is "
    "set to 0. TaskGraphs lower than this critical path duration will not be released.",
)
flags.DEFINE_list(
    "alibaba_loader_max_critical_path_runtimes",
    [],
    "The maximum critical path duration for each TaskGraph from the corresponding "
    "Workload. If the list is empty, then the maximum critical path duration is "
    "set to the maximum critical path duration of the Workload. TaskGraphs higher "
    "than this critical path duration will not be released.",
)
flags.DEFINE_bool(
    "alibaba_enable_heterogeneous_resource_type",
    False,
    "If true, we use heterogeneous resource types with %difference in runtime.",
)
flags.DEFINE_bool(
    "alibaba_dump_filtered_dags",
    False,
    "If True, we dump the filtered DAGs into a separate file to speed up processing.",
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
    "min_deadline",
    0,
    "The minimum deadline to assign to a task (in µs).",
)
flags.DEFINE_integer(
    "min_deadline_variance",
    0,
    "The minimum % variance to allocate to the assigned deadline for each task.",
)
flags.DEFINE_list(
    "min_deadline_variances",
    [],
    "The minimum % variance to allocate to the assigned deadline for each TaskGraph "
    "from the corresponding Workload. If only one Workload profile is provided, then "
    "use `min_deadline_variance`.",
)
flags.DEFINE_integer(
    "max_deadline",
    sys.maxsize,
    "The maximum deadline to assign to a task (in µs).",
)
flags.DEFINE_integer(
    "max_deadline_variance",
    20,
    "The maximum % variance to allocate to the assigned deadline for each task.",
)
flags.DEFINE_list(
    "max_deadline_variances",
    [],
    "The maximum % variance to allocate to the assigned deadline for each TaskGraph "
    "from the corresponding Workload. If only one Workload profile is provided, then "
    "use `max_deadline_variance`.",
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
flags.DEFINE_bool(
    "release_taskgraphs",
    False,
    "If True, all tasks from a graph are released if any of the tasks have "
    "reached their release time.",
)

# Scheduler related flags.
flags.DEFINE_enum(
    "scheduler",
    "EDF",
    [
        "FIFO",
        "EDF",
        "LSF",
        "Z3",
        "BranchPrediction",
        "ILP",
        "TetriSched_CPLEX",
        "TetriSched_Gurobi",
        "Clockwork",
        "TetriSched",
        "GraphenePrime",
        "Graphene",
    ],
    "The scheduler to use for this execution.",
)
flags.DEFINE_bool(
    "verify_schedule",
    False,
    "If True, the schedule returned by the scheduler is verified after every call, and"
    " an error is raised if a consistency check fails.",
)
flags.DEFINE_bool(
    "preemption", False, "Enable preemption of running tasks in the scheduler."
)
flags.DEFINE_bool(
    "retract_schedules", False, "Enable the retraction of previously decided schedules."
)
flags.DEFINE_integer(
    "scheduler_runtime",
    -1,
    "The runtime (in µs) to assign to each scheduler invocation.",
)
flags.DEFINE_integer(
    "scheduler_frequency",
    -1,
    "The minimum time interval (in µs) to elapse between too scheduler invocations.",
)
flags.DEFINE_bool(
    "scheduler_run_at_worker_free",
    False,
    "If `True`, the scheduler is invoked whenever a Task finishes completion and a "
    "Worker becomes free. This is a way of specifying a coarse-grained scheduler "
    "frequency that is not tied to the release of Tasks in the system.",
)
flags.DEFINE_bool(
    "scheduler_adaptive_discretization",
    False,
    "If `True`, the scheduler creates space-time matrix non-uniformly. "
    "The discretization is finer initially, and coarser at the end. (default: False)",
)
# flags.DEFINE_bool(
#     "scheduler_dynamic_discretization",
#     False,
#     "If `True`, the scheduler creates space-time matrix non-uniformly. "
#     "The discretization is dynamically decided based on the occupancy request for "
#     "each time slice. (default: False)",
# )
flags.DEFINE_bool(
    "finer_discretization_at_prev_solution",
    False,
    "If `True`, the scheduler keeps discretization of 1 around previous solution. "
    "The discretization is dynamically decided based on the occupancy request for "
    "each time slice. (default: False)",
)
flags.DEFINE_integer(
    "finer_discretization_window",
    5,
    "The window around previous solution that keeps discretization of 1.",
)
flags.DEFINE_integer(
    "scheduler_max_time_discretization",
    5,
    "The maximum discretization that the scheduler can have (in µs). "
    "Only used when scheduler_adaptive_discretization flag is enabled. (default: 5)",
)
flags.DEFINE_float(
    "scheduler_max_occupancy_threshold",
    0.8,
    "The percentage b/w 0 and 1 of maximum occupancy beyond which the discretization "
    "would always be 1 incase of dynamic discretization. "
    "This flag is only used when dynamic discretization is enabled (default: 0.8)",
)
flags.DEFINE_integer(
    "scheduler_delay",
    0,
    "The delay (in µs) associated with invoking a scheduler after the "
    "release of a Task in the system.",
)
flags.DEFINE_integer(
    "scheduler_lookahead",
    0,
    "The scheduler places tasks that are estimated to be released "
    "within the scheduling lookahead (in µs).",
)
flags.DEFINE_integer(
    "scheduler_plan_ahead",
    -1,
    "The length of the space-time matrix to construct in the future to consider packing"
    "the available tasks (in µs). The default value is to pack until the maximum "
    "deadline for each batch of available tasks.",
)
flags.DEFINE_integer(
    "scheduler_plan_ahead_no_consideration_gap",
    0,
    "The length of time gap (in µs) for which the reconsiderations are frozen. "
    "From the current time to the consideration gap, any tasks placed will not be "
    "reconsidered for rescheduling.",
)
flags.DEFINE_integer(
    "scheduler_time_discretization",
    1,
    "The length of each slot in the space-time matrix to consider for scheduling the "
    "tasks (in µs). The default value is 1µs, and a higher value can lead to faster "
    "solutions but a potentially lower goodput due to resources being blocked for the "
    "entirety of the slot.",
)
flags.DEFINE_enum(
    "scheduler_policy",
    "worst",
    ["best", "worst", "max", "random"],
    "The policy to be used for the BranchPredictionScheduler.",
)
flags.DEFINE_float(
    "branch_prediction_accuracy",
    0.5,
    "The probability with which to correctly guess the branch that is to be taken.",
)
flags.DEFINE_bool(
    "enforce_deadlines",
    False,
    "True if the ILP formulation must ensure that deadlines are met.",
)
flags.DEFINE_enum(
    "ilp_goal",
    "max_goodput",
    ["max_slack", "min_placement_delay", "max_goodput"],
    "Sets the goal of the mathematical optimization-backed solvers.",
)
flags.DEFINE_enum(
    "clockwork_goal",
    "clockwork",
    ["clockwork", "least_slack"],
    "Sets the goal of the Clockwork scheduler.",
)
flags.DEFINE_integer(
    "scheduler_time_limit",
    -1,
    "The time limit (in seconds) to allow the scheduler to keep "
    "searching for solutions without finding a better one.",
)
flags.DEFINE_bool(
    "scheduler_log_to_file",
    False,
    "If True, the scheduler writes the status of each run to a seperate"
    "log file in a format unique to every scheduler.",
)
flags.DEFINE_bool(
    "decompose_deadlines",
    False,
    "If True, the task deadline is decided by decomposing the TaskGraph's deadline "
    "according to the critical path of the TaskGraph.",
)
flags.DEFINE_list(
    "scheduler_log_times",
    [],
    "A list of timestamps (in µs) at which to request extra logging from the Scheduler."
    "If scheduler_log_to_file is `True`, then extra information will be requested for "
    "all timestamps.",
)
flags.DEFINE_bool(
    "scheduler_run_load",
    False,
    "If `True`, the scheduler is allowed to make LOAD_PROFILE "
    "and EVICT_PROFILE actions.",
)
flags.DEFINE_bool(
    "scheduler_enable_batching",
    False,
    "If `True`, the scheduler is allowed to batch tasks "
    "that share a WorkProfile together.",
)
flags.DEFINE_bool(
    "scheduler_selective_rescheduling",
    False,
    "If `True`, the supported schedulers will follow some pre-defined strategies for "
    "selectively sampling TaskGraphs to reschedule.",
)
flags.DEFINE_integer(
    "scheduler_selective_rescheduling_sample_size",
    5,
    "If `scheduler_selective_rescheduling` is True, then this flag defines the number "
    "of TaskGraphs to sample for rescheduling.",
)
flags.DEFINE_float(
    "scheduler_reconsideration_period",
    0.1,
    "The percentage of critical path duration until which the scheduler will try "
    "placing the TaskGraph, and drop the TaskGraph if it cannot be placed after.",
)
flags.DEFINE_multi_enum(
    "optimization_passes",
    [],
    [
        "CRITICAL_PATH_PASS",
        "CAPACITY_CONSTRAINT_PURGE_PASS",
        "DYNAMIC_DISCRETIZATION_PASS",
    ],
    "Specify the optimizations that needs to be enabled once the STRL is generated.",
)

# Workload definition related flags.
flags.DEFINE_integer(
    "randomize_start_time_min",
    0,
    "Choose a random start time for the Workload release from this range (in µs).",
)
flags.DEFINE_integer(
    "randomize_start_time_max",
    0,
    "Choose a random start time for the Workload release from this range (in µs).",
)
flags.DEFINE_enum(
    "override_release_policy",
    "fixed",
    JobGraph.RELEASE_POLICIES,
    "Override the release policy for all TaskGraphs defined in the Workload.",
)
flags.DEFINE_list(
    "override_release_policies",
    [],
    "Override the release policy for all TaskGraphs defined in each Workload."
    "If provided, the list must be of the same length as the list of workload "
    "profile paths. For a single workload profile path, use `override_release_policy`.",
)
flags.DEFINE_integer(
    "override_num_invocation",
    0,
    "Override the number of invocations for all TaskGraphs defined in the Workload.",
)
flags.DEFINE_list(
    "override_num_invocations",
    [],
    "Override the number of invocations for all TaskGraphs defined in each Workload."
    "If provided, the list must be of the same length as the list of workload "
    "profile paths. For a single workload profile path, use `override_num_invocation`.",
)
flags.DEFINE_float(
    "override_poisson_arrival_rate",
    0.0,
    "Override the Poisson arrival rate for all TaskGraphs defined"
    "in the JSON workload definition.",
)
flags.DEFINE_list(
    "override_poisson_arrival_rates",
    [],
    "Override the Poisson arrival rate for all TaskGraphs defined in each Workload."
    "If provided, the list must be of the same length as the list of workload "
    "profile paths. For a single workload profile path, use "
    "`override_poisson_arrival_rate`.",
)
flags.DEFINE_float(
    "override_base_arrival_rate",
    0.0,
    "Override the base arrival rate for all TaskGraphs defined"
    "in the JSON workload definition.",
)
flags.DEFINE_float(
    "override_gamma_coefficient",
    0.0,
    "Override the coefficient of variation in a Gamma distribution for all "
    "TaskGraphs defined in the JSON workload definition.",
)
flags.DEFINE_integer(
    "override_arrival_period",
    0,
    "Override the arrival period for all Taskgraphs defined in "
    "the JSON workload definition.",
)
flags.DEFINE_integer(
    "override_slo",
    -1,
    "Override the SLO for all TaskGraphs defined in the JSON workload definition."
    "If this is not set, the deadline is inferred from the task's execution strategy.",
)
flags.DEFINE_bool(
    "unique_work_profiles",
    False,
    "If True, then the same WorkProfile is shared by multiple JobGraphs.",
)
flags.DEFINE_integer(
    "replication_factor",
    1,
    "The number of times to replicate each JobGraph in the Workload definition."
    "Set --unique_work_profiles to True to ensure that the same WorkProfile is not "
    "used for multiple JobGraphs.",
)


def main(args):
    """Main loop that loads the data from the given profile paths, and
    runs the Simulator on the data with the given scheduler.
    """
    if FLAGS.log_file_mode == "write":
        # Delete the prior log file if it exists.
        if FLAGS.log_file_name is not None and os.path.exists(FLAGS.log_file_name):
            os.remove(FLAGS.log_file_name)
        if FLAGS.csv_file_name is not None and os.path.exists(FLAGS.csv_file_name):
            os.remove(FLAGS.csv_file_name)

    random.seed(FLAGS.random_seed)
    logger = setup_logging(
        name=__name__,
        log_dir=FLAGS.log_dir,
        log_file=FLAGS.log_file_name,
        log_level=FLAGS.log_level,
    )
    logger.info("Starting the execution of the simulator loop.")
    logger.info("Workload File: %s", FLAGS.workload_profile_path)
    logger.info("Workers File: %s", FLAGS.worker_profile_path)

    csv_logger = setup_csv_logging(
        name=__name__,
        log_dir=FLAGS.log_dir,
        log_file=FLAGS.csv_file_name,
    )
    for flag_name in FLAGS:
        csv_logger.debug(f"input_flag,{flag_name},{getattr(FLAGS, flag_name)}")

    # Load the data.
    if FLAGS.execution_mode == "replay":
        if FLAGS.replay_trace == "pylot":
            workload_loader = WorkloadLoader(
                path=FLAGS.workload_profile_path, _flags=FLAGS
            )
            job_graph = workload_loader.workload.get_job_graph("pylot_dataflow")
            task_loader = TaskLoaderPylot(
                job_graph=job_graph,
                graph_name="pylot_dataflow",
                profile_path=FLAGS.workload_profile_path,
                _flags=FLAGS,
            )
            workload = Workload.from_task_graphs(  # noqa: F841
                {"pylot_dataflow": task_loader.get_task_graph()},
                _flags=FLAGS,
            )
            raise NotImplementedError(
                "Pylot loader does not yet support dynamic workloads."
            )
        elif FLAGS.replay_trace == "clockwork_bursty":
            workload_loader = WorkloadLoaderClockworkBursty()
            raise NotImplementedError(
                "Clockwork loader does not yet support dynamic workloads."
            )
        elif FLAGS.replay_trace == "alibaba":
            workload_loader = AlibabaLoader(
                workload_interval=EventTime(
                    FLAGS.workload_update_interval, EventTime.Unit.US
                ),
                flags=FLAGS,
            )
        elif FLAGS.replay_trace == "tpch":
            workload_loader = TpchLoader(
                path=FLAGS.tpch_query_dag_spec,
                flags=FLAGS,
            )
        else:
            raise NotImplementedError(
                f"Replay trace {FLAGS.replay_trace} is not implemented yet."
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
    elif FLAGS.execution_mode == "json" or FLAGS.execution_mode == "yaml":
        workload_loader = WorkloadLoader(path=FLAGS.workload_profile_path, _flags=FLAGS)

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

    # Retrieve the branch prediction policy from the flags.
    if FLAGS.scheduler_policy == "best":
        branch_prediction_policy = BranchPredictionPolicy.BEST_CASE
    elif FLAGS.scheduler_policy == "worst":
        branch_prediction_policy = BranchPredictionPolicy.WORST_CASE
    elif FLAGS.scheduler_policy == "max":
        branch_prediction_policy = BranchPredictionPolicy.MAXIMUM
    elif FLAGS.scheduler_policy == "random":
        branch_prediction_policy = BranchPredictionPolicy.RANDOM
    else:
        raise NotImplementedError(
            f"The policy {FLAGS.scheduler_policy} is not supported."
        )

    # Instantiate the scheduler based on the given flag.
    scheduler = None
    if FLAGS.scheduler == "FIFO":
        from schedulers import FIFOScheduler

        scheduler = FIFOScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "EDF":
        from schedulers import EDFScheduler

        scheduler = EDFScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "LSF":
        from schedulers import LSFScheduler

        scheduler = LSFScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "Z3":
        from schedulers import Z3Scheduler

        scheduler = Z3Scheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            policy=branch_prediction_policy,
            branch_prediction_accuracy=FLAGS.branch_prediction_accuracy,
            retract_schedules=FLAGS.retract_schedules,
            release_taskgraphs=FLAGS.release_taskgraphs,
            goal=FLAGS.ilp_goal,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "BranchPrediction":
        from schedulers import BranchPredictionScheduler

        scheduler = BranchPredictionScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            policy=branch_prediction_policy,
            branch_prediction_accuracy=FLAGS.branch_prediction_accuracy,
            release_taskgraphs=FLAGS.release_taskgraphs,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "ILP":
        from schedulers import ILPScheduler

        scheduler = ILPScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            policy=branch_prediction_policy,
            branch_prediction_accuracy=FLAGS.branch_prediction_accuracy,
            retract_schedules=FLAGS.retract_schedules,
            release_taskgraphs=FLAGS.release_taskgraphs,
            goal=FLAGS.ilp_goal,
            batching=FLAGS.scheduler_enable_batching,
            time_limit=EventTime(FLAGS.scheduler_time_limit, EventTime.Unit.S),
            log_to_file=FLAGS.scheduler_log_to_file,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "TetriSched_CPLEX":
        from schedulers import TetriSchedCPLEXScheduler

        scheduler = TetriSchedCPLEXScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            retract_schedules=FLAGS.retract_schedules,
            goal=FLAGS.ilp_goal,
            batching=FLAGS.scheduler_enable_batching,
            time_limit=EventTime(FLAGS.scheduler_time_limit, EventTime.Unit.S),
            time_discretization=EventTime(
                FLAGS.scheduler_time_discretization, EventTime.Unit.US
            ),
            plan_ahead=EventTime(FLAGS.scheduler_plan_ahead, EventTime.Unit.US),
            log_to_file=FLAGS.scheduler_log_to_file,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "TetriSched_Gurobi":
        from schedulers import TetriSchedGurobiScheduler

        scheduler = TetriSchedGurobiScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            retract_schedules=FLAGS.retract_schedules,
            release_taskgraphs=FLAGS.release_taskgraphs,
            goal=FLAGS.ilp_goal,
            batching=FLAGS.scheduler_enable_batching,
            time_limit=EventTime(FLAGS.scheduler_time_limit, EventTime.Unit.S),
            time_discretization=EventTime(
                FLAGS.scheduler_time_discretization, EventTime.Unit.US
            ),
            plan_ahead=EventTime(FLAGS.scheduler_plan_ahead, EventTime.Unit.US),
            log_to_file=FLAGS.scheduler_log_to_file,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "Clockwork":
        from schedulers import ClockworkScheduler

        scheduler = ClockworkScheduler(
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            goal=FLAGS.clockwork_goal,
            _flags=FLAGS,
        )
    elif FLAGS.scheduler == "TetriSched":
        from schedulers import TetriSchedScheduler

        finer_discretization = FLAGS.finer_discretization_at_prev_solution
        scheduler = TetriSchedScheduler(
            preemptive=FLAGS.preemption,
            runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
            lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
            enforce_deadlines=FLAGS.enforce_deadlines,
            retract_schedules=FLAGS.retract_schedules,
            release_taskgraphs=FLAGS.release_taskgraphs,
            goal=FLAGS.ilp_goal,
            time_discretization=EventTime(
                FLAGS.scheduler_time_discretization, EventTime.Unit.US
            ),
            plan_ahead=EventTime(FLAGS.scheduler_plan_ahead, EventTime.Unit.US),
            log_to_file=FLAGS.scheduler_log_to_file,
            adaptive_discretization=FLAGS.scheduler_adaptive_discretization,
            _flags=FLAGS,
            max_time_discretization=EventTime(
                FLAGS.scheduler_max_time_discretization, EventTime.Unit.US
            ),
            max_occupancy_threshold=FLAGS.scheduler_max_occupancy_threshold,
            finer_discretization_at_prev_solution=finer_discretization,
            finer_discretization_window=EventTime(
                FLAGS.finer_discretization_window, EventTime.Unit.US
            ),
            plan_ahead_no_consideration_gap=EventTime(
                FLAGS.scheduler_plan_ahead_no_consideration_gap, EventTime.Unit.US
            ),
        )
    elif FLAGS.scheduler == "GraphenePrime":
        try:
            from schedulers import TetriSchedScheduler

            scheduler = TetriSchedScheduler(
                preemptive=FLAGS.preemption,
                runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
                lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
                # Graphene does not have a notion of deadlines.
                enforce_deadlines=False,
                retract_schedules=FLAGS.retract_schedules,
                # Graphene is a DAG-aware scheduler.
                release_taskgraphs=True,
                # Graphene is a min-makespan scheduler.
                goal="min_placement_delay",
                time_discretization=EventTime(
                    FLAGS.scheduler_time_discretization, EventTime.Unit.US
                ),
                plan_ahead=EventTime(FLAGS.scheduler_plan_ahead, EventTime.Unit.US),
                log_to_file=FLAGS.scheduler_log_to_file,
                _flags=FLAGS,
            )
        except ImportError:
            logger.error(
                "Unable to import TetriSchedScheduler. "
                "Make sure you've compiled the TetriSched C++ backend."
            )
            raise RuntimeError("TetriSchedScheduler not available.")
    elif FLAGS.scheduler == "Graphene":
        try:
            from schedulers import GrapheneScheduler

            scheduler = GrapheneScheduler(
                preemptive=FLAGS.preemption,
                runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
                lookahead=EventTime(FLAGS.scheduler_lookahead, EventTime.Unit.US),
                retract_schedules=FLAGS.retract_schedules,
                goal=FLAGS.ilp_goal,
                time_discretization=EventTime(
                    FLAGS.scheduler_time_discretization, EventTime.Unit.US
                ),
                plan_ahead=EventTime(FLAGS.scheduler_plan_ahead, EventTime.Unit.US),
                log_to_file=FLAGS.scheduler_log_to_file,
                _flags=FLAGS,
            )
        except ImportError:
            logger.error(
                "Unable to import GrapheneScheduler. "
                "Make sure you've compiled the TetriSched C++ backend."
            )
            raise RuntimeError("GrapheneScheduler not available.")
    else:
        raise ValueError(
            "Unsupported scheduler implementation: {}".format(FLAGS.scheduler)
        )

    # Load the worker topology.
    if FLAGS.execution_mode in ["replay", "synthetic", "json", "yaml"]:
        worker_loader = WorkerLoader(
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
        workload_loader=workload_loader,
        loop_timeout=EventTime(FLAGS.loop_timeout, EventTime.Unit.US),
        scheduler_frequency=EventTime(FLAGS.scheduler_frequency, EventTime.Unit.US),
        _flags=FLAGS,
    )
    if FLAGS.dry_run:
        simulator.dry_run()
    else:
        simulator.simulate()


if __name__ == "__main__":
    app.run(main)
