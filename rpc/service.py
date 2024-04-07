import asyncio
import heapq
import math
import os
import sys
import time
from collections import defaultdict
from concurrent import futures
from operator import attrgetter
from typing import Mapping, Sequence
from urllib.parse import urlparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import erdos_scheduler_pb2
import erdos_scheduler_pb2_grpc
import grpc
from absl import app, flags
from tpch_utils import get_all_stage_info_for_query, verify_and_relable_tpch_app_graph

from schedulers import EDFScheduler, FIFOScheduler, TetriSchedScheduler
from utils import EventTime, setup_logging
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Placement,
    Resource,
    Resources,
    Task,
    TaskGraph,
    Workload,
    WorkProfile,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 50051, "Port to serve the ERDOS Scheduling RPC Server on.")
flags.DEFINE_integer(
    "max_workers", 10, "Maximum number of workers to use for the RPC server."
)
flags.DEFINE_string("log_file_name", None, "Name of the log file.", short_name="log")
flags.DEFINE_string("log_level", "debug", "The level to log.")
flags.DEFINE_integer(
    "initial_executors",
    10,
    "The initial number of executors that are requested by each application.",
)
flags.DEFINE_float(
    "spark_task_duration_multiplier",
    1,
    "The multiplier used for spark job task runtimes. Buffer time is added "
    "to ensure that tasks complete before the scheduler expects it to complete. "
    "Completion of tasks after the scheduler's expected task completion time "
    "is detrimental for scheduler's planning and could invalidate some schedules",
)
flags.DEFINE_integer(
    "virtualized_cores",
    500,
    "The number of virtualized cores that must be created in each Worker on the "
    "framework. This allows us to spawn a higher number of executors than the number "
    "possible with actual available resources. Thus, we can spawn the executors for "
    "each application, and only selectively activate them according to the actual "
    "available resources.",
)
flags.DEFINE_integer(
    "virtualized_memory",
    500,
    "The amount of virtualized memory (in GB) that must be created in each Worker on "
    "the framework. Refer to the `virtualized_cores` flag for more information.",
)
flags.DEFINE_enum(
    "scheduler", "DAGSched", ["FIFO", "EDF", "DAGSched"], "The scheduler to use for "
    "this execution."
)
flags.DEFINE_enum(
    "tpch_profile_type", "Cloudlab", ["Decima", "Cloudlab"], "The set of profiles to "
    "use for execution of tpch queries. Note that Cloudlab profile has all 22 queries. "
    "From the Decima profile we support only 15 queries (1-10, 12-14, 16, 19). The "
    "rest might also run but DAG structure might not match Decima profiles."
)
flags.DEFINE_enum(
    "tpch_dataset_size", "50g", ["50g", "100g", "250g", "500g"], "Options for "
    "dataset size of TPCH query. The Cloudlab profile will be picked accordingly. "
)
flags.DEFINE_enum(
    "tpch_max_executors_per_job", "50", ["50", "75", "100", "200"], "Options for "
    "max executors to use for tpch queries. The Cloudlab profile will be picked "
    "accordingly."
)
flags.DEFINE_bool(
    "override_worker_cpu_count",
    False,
    "If True, worker CPU count will be set to INT_MAX. This allows us to scale up "
    "spark experiments without actually deploying a large spark cluster.",
)
flags.DEFINE_bool(
    "use_profile_to_scale_executors",
    False,
    "If True, it means that a fixed number of (max) executors was given to the "
    "spark job to run. With this profile, we can directly use the profiled "
    "stage runtime, while setting the number of required slots or executors "
    "to 1 per stage. This allows us do the same scheduling but creates less "
    "overhead for this rpc service while running the experiments.",
)
flags.DEFINE_bool(
    "release_taskgraphs",
    False,
    "If True, all tasks from a graph are released if any of the tasks have "
    "reached their release time.",
)
flags.DEFINE_bool(
    "enforce_deadlines",
    False,
    "True if the ILP formulation must ensure that deadlines are met.",
)
flags.DEFINE_integer(
    "scheduler_time_discretization",
    1,
    "The length of each slot in the space-time matrix to consider for scheduling the "
    "tasks (in µs). The default value is 1µs, and a higher value can lead to faster "
    "solutions but a potentially lower goodput due to resources being blocked for the "
    "entirety of the slot.",
)
flags.DEFINE_bool(
    "scheduler_enable_optimization_pass",
    False,
    "If `True`, the scheduler runs pre/post-translation optimization passes"
    "when registering STRL expression.",
)
flags.DEFINE_float(
    "scheduler_reconsideration_period",
    0.1,
    "The percentage of critical path duration until which the scheduler will try "
    "placing the TaskGraph, and drop the TaskGraph if it cannot be placed after.",
)
flags.DEFINE_bool(
    "retract_schedules", False, "Enable the retraction of previously decided schedules."
)
flags.DEFINE_integer(
    "scheduler_time_limit",
    -1,
    "The time limit (in seconds) to allow the scheduler to keep "
    "searching for solutions without finding a better one.",
)
flags.DEFINE_bool(
    "scheduler_dynamic_discretization",
    False,
    "If `True`, the scheduler creates space-time matrix non-uniformly. "
    "The discretization is dynamically decided based on the occupancy request for "
    "each time slice. (default: False)",
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
flags.DEFINE_bool(
    "scheduler_selective_rescheduling",
    False,
    "If `True`, the supported schedulers will follow some pre-defined strategies for "
    "selectively sampling TaskGraphs to reschedule.",
)
flags.DEFINE_integer(
    "scheduler_plan_ahead_no_consideration_gap",
    10,
    "The length of time gap (in µs) for which the reconsiderations are frozen. "
    "From the current time to the consideration gap, any tasks placed will not be "
    "reconsidered for rescheduling.",
)
flags.DEFINE_list(
    "scheduler_log_times",
    [],
    "A list of timestamps (in µs) at which to request extra logging from the Scheduler."
    "If scheduler_log_to_file is `True`, then extra information will be requested for "
    "all timestamps.",
)
flags.DEFINE_integer(
    "scheduler_selective_rescheduling_sample_size",
    5,
    "If `scheduler_selective_rescheduling` is True, then this flag defines the number "
    "of TaskGraphs to sample for rescheduling.",
)


# Define an item containing completion timestamp and task
class TimedItem:
    _next_id = 0
    _id_threshold = 99999

    def __init__(self, timestamp, task):
        self.timestamp = timestamp
        self.task = task
        self.id = TimedItem._next_id
        TimedItem._next_id += 1

        # Reset _next_id if it crosses the threshold
        # We keep _next_id bounded to avoid very large numbers
        # which could lead to slightly slower comparions
        if TimedItem._next_id > TimedItem._id_threshold:
            TimedItem._next_id = 0

    def __lt__(self, other):
        """Less than comparison for TimedItem instances."""
        if self.timestamp == other.timestamp:
            # Unique ID for each TimedItem acts as tie-breaker
            # for inserting into PriorityQueue
            return self.id < other.id
        return self.timestamp < other.timestamp

    def __eq__(self, other):
        """Equality comparison for TimedItem instances."""
        return self.timestamp == other.timestamp and self.id == other.id


# Define a priority queue based on heapq module
class PriorityQueue:
    def __init__(self):
        self._queue = []

    def put(self, item):
        heapq.heappush(self._queue, (item.timestamp, item))

    def get(self):
        _, item = heapq.heappop(self._queue)
        return item

    def empty(self):
        return len(self._queue) == 0


# Implement the service.
class SchedulerServiceServicer(erdos_scheduler_pb2_grpc.SchedulerServiceServicer):
    def __init__(self) -> None:
        """Initialize the service, and setup the logger."""
        # Values used by the Servicer.
        self._logger = setup_logging(name=FLAGS.log_file_name,
                                     log_level=FLAGS.log_level)
        # self._logger = setup_logging(
        #     name=__name__,
        #     log_dir=FLAGS.log_dir,
        #     log_file=FLAGS.log_file_name,
        #     log_level=FLAGS.log_level
        #     )
        self._initialized = False
        self._initialization_time = -1
        self._master_uri = None

        # The simulator types maintained by the Servicer.
        self._worker_pool = None
        self._worker_pools = None
        self._drivers: Mapping[str, Task] = {}
        self._workload = None

        # Scheduler information maintained by the servicer.
        self._scheduler_running_lock = asyncio.Lock()
        self._scheduler_running = False
        self._rerun_scheduler = False
        if FLAGS.scheduler == "EDF":
            self._scheduler = EDFScheduler(
                enforce_deadlines=FLAGS.enforce_deadlines,
            )
        elif FLAGS.scheduler == "FIFO":
            self._scheduler = FIFOScheduler(
                enforce_deadlines=FLAGS.enforce_deadlines,
            )
        elif FLAGS.scheduler == "DAGSched":
            # --scheduler=TetriSched
            # --release_taskgraphs
            # --enforce_deadlines
            # --scheduler_time_discretization=1 ====> Conv to EventTime & passed through diff arg name
            # --scheduler_enable_optimization_pass ====> Passed through _flags
            # --retract_schedules
            # --scheduler_dynamic_discretization ====> Passed through different argument name
            # --scheduler_max_time_discretization=8 ====> Conv to EventTime & passed through diff arg name
            # --scheduler_max_occupancy_threshold=0.999 ====> Passed through different argument name
            # --finer_discretization_at_prev_solution
            # --finer_discretization_window=4
            # --scheduler_selective_rescheduling ====> Passed through _flags
            # --scheduler_reconsideration_period=0.6 ====> Passed through _flags

            self._scheduler = TetriSchedScheduler(
                release_taskgraphs=FLAGS.release_taskgraphs,
                time_discretization=EventTime(
                    FLAGS.scheduler_time_discretization, EventTime.Unit.US
                ),
                _flags=FLAGS,
                max_time_discretization=EventTime(
                    FLAGS.scheduler_max_time_discretization, EventTime.Unit.US
                    ),
                enforce_deadlines=FLAGS.enforce_deadlines,
                dynamic_discretization=FLAGS.scheduler_dynamic_discretization,
                max_occupancy_threshold=FLAGS.scheduler_max_occupancy_threshold,
                retract_schedules=FLAGS.retract_schedules,
                finer_discretization_at_prev_solution=(
                    FLAGS.finer_discretization_at_prev_solution
                ),
                finer_discretization_window=EventTime(
                    FLAGS.finer_discretization_window, EventTime.Unit.US
                    ),
                plan_ahead_no_consideration_gap=EventTime(
                    FLAGS.scheduler_plan_ahead_no_consideration_gap, EventTime.Unit.US
                    ),
            )

        else:
            raise ValueError(f"Unknown scheduler {FLAGS.scheduler}.")

        # Placement information maintained by the servicer.
        # The placements map the application IDs to the Placement retrieved from the
        # scheduler. The placements are automatically clipped at the time of informing
        # the framework of applying them to the executors.
        # NOTE (Sukrit): This must always be sorted by the Placement time.
        self._placements: Mapping[str, Sequence[Placement]] = defaultdict(list)

        # Additional task information maintained by the servicer
        self._tasks_marked_for_completion = PriorityQueue()

        # Start the asyncio loop for clearing out pending tasks for completion
        asyncio.create_task(self.PopTasksBasedOnTime())

        super().__init__()

    async def schedule(self) -> None:
        """Schedules the tasks that have been added to the Workload."""
        async with self._scheduler_running_lock:
            if self._scheduler_running:
                self._logger.error(
                    "Scheduler already running, this should never be reached."
                )
                return
            self._scheduler_running = True

        current_time = EventTime(int(time.time()), EventTime.Unit.US)
        self._logger.info(
            "Starting a scheduling cycle with %s TaskGraphs and %s Workers at %s.",
            len(self._workload.task_graphs),
            len(self._worker_pool.workers),
            current_time,
        )

        # TODO (Sukrit): Change this to a better implementation.
        # Let's do some simple scheduling for now, that gives a fixed number of
        # executors to all the available applications in intervals of 10 seconds.
        if len(self._workload.task_graphs) >= 2:
            placements = self._scheduler.schedule(
                sim_time=EventTime(current_time.time, EventTime.Unit.US),
                workload=self._workload,
                worker_pools=self._worker_pools,
            )
            # Filter the placements that are not of type PLACE_TASK and that have not
            # been placed.
            filtered_placements = filter(
                lambda p: p.placement_type == Placement.PlacementType.PLACE_TASK
                and p.is_placed(),
                placements,
            )
            for placement in sorted(
                filtered_placements, key=attrgetter("placement_time")
            ):
                self._placements[placement.task.task_graph].append(placement)
                # Schedule the task here since marking it as running requires it to be
                # scheduled before. We mark it to be running when we inform the
                # framework of the placement.
                placement.task.schedule(
                    time=placement.placement_time,
                    placement=placement,
                )

        self._logger.info(
            "Finished the scheduling cycle initiated at %s.", current_time
        )

        # Check if another run of the Scheduler has been requested, and if so, create
        # a task for it. Otherwise, mark the scheduler as not running.
        async with self._scheduler_running_lock:
            self._scheduler_running = False
            if self._rerun_scheduler:
                self._rerun_scheduler = False
                asyncio.create_task(self.schedule())

    async def run_scheduler(self) -> None:
        """Checks if the scheduler is running, and if not, starts it.

        If the scheduler is already running, we queue up another execution of the
        scheduler. This execution batches the scheduling requests, and runs the
        scheduler only once for all the requests."""
        async with self._scheduler_running_lock:
            if not self._scheduler_running:
                asyncio.create_task(self.schedule())
            else:
                self._rerun_scheduler = True

    async def RegisterFramework(self, request, context):
        """Registers a new framework with the backend scheduler.
        This is the entry point for a new instance of Spark / Flink to register
        itself with the backend scheduler, and is intended as an EHLO.
        """
        if self._initialized:
            self._logger.warning(
                "Framework already registered at %s with the address %s",
                self._initialization_time,
                self._master_uri,
            )
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=f"Framework already registered at "
                f"{self._initialization_time} at the address {self._master_uri}",
            )

        # Setup a new Framework instance.
        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = request.timestamp
        self._initialized = True
        self._logger.info(
            "Registering framework %s with URI %s at %s",
            framework_name,
            self._master_uri,
            self._initialization_time,
        )

        # Setup the simulator types.
        parsed_uri = urlparse(self._master_uri)
        self._worker_pool = WorkerPool(name=f"WorkerPool_{parsed_uri.netloc}")
        self._worker_pools = WorkerPools(worker_pools=[self._worker_pool])
        self._workload = Workload.from_task_graphs({})

        # Return the response.
        return erdos_scheduler_pb2.RegisterFrameworkResponse(
            success=True,
            message=f"{framework_name} at {self._master_uri} registered successfully!",
        )

    async def RegisterDriver(self, request, context):
        if not self._initialized:
            self._logger.warning(
                "Trying to register a driver with name %s and id %s, "
                "but no framework is registered yet.",
                request.name,
                request.id,
            )
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message="Framework not registered yet.",
                worker_id="",
            )

        # Create a Task for the Driver, and add it to the list of drivers.
        # TODO (Sukrit): We drop the memory requirements for now, we should use
        # them to do multi-dimensional packing using STRL.
        self._logger.info(
            "Received a request to register a driver with name %s, URI: %s. "
            "The driver requires %s cores and %s memory.",
            request.id,
            request.uri,
            request.cores,
            request.memory,
        )
        driver_resources = Resources(
            resource_vector={Resource(name="Slot_CPU", _id="any"): 1}
        )
        driver_job = Job(
            name=request.id,
            profile=WorkProfile(
                name=f"WorkProfile_{request.id}",
                execution_strategies=ExecutionStrategies(
                    [
                        ExecutionStrategy(
                            resources=driver_resources,
                            batch_size=1,
                            # NOTE (Sukrit): Drivers are long running, and have no
                            # fixed runtime. Setting it to zero helps us unload the
                            # driver from the Worker whenever we need it.
                            runtime=EventTime.zero(),
                        )
                    ]
                ),
            ),
        )
        driver = Task(
            name=request.id,
            task_graph=request.uri,
            job=driver_job,
            deadline=EventTime.invalid(),
        )
        self._drivers[request.id] = driver

        # Iterate over the Workers and find a Worker that can accomodate the driver.
        placement_found = False
        for worker in self._worker_pool.workers:
            for execution_strategy in driver.available_execution_strategies:
                if worker.can_accomodate_strategy(execution_strategy):
                    # This Worker can accomodate the Driver, we assign it here.
                    placement_found = True
                    self._worker_pool.place_task(driver, execution_strategy, worker.id)

                    # Update the Task's state and placement information.
                    placement_time = EventTime(request.timestamp, EventTime.Unit.US)
                    driver.schedule(
                        time=placement_time,
                        placement=Placement(
                            type=Placement.PlacementType.PLACE_TASK,
                            computation=driver,
                            placement_time=placement_time,
                            worker_pool_id=self._worker_pool.id,
                            worker_id=worker.id,
                            strategy=execution_strategy,
                        ),
                    )
                    driver.start(placement_time)

                    # Tell the framework to start the driver.
                    return erdos_scheduler_pb2.RegisterDriverResponse(
                        success=True,
                        message=f"Driver {request.id} registered successfully!",
                        worker_id=worker.name,
                    )

        if not placement_found:
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message=f"No Worker can accomodate the driver {request.id} yet.",
                worker_id="",
            )

    async def DeregisterDriver(self, request, context):
        if not self._initialized:
            self._logger.warning(
                "Trying to deregister a driver with id %s, "
                "but no framework is registered yet.",
                request.id,
            )
            return erdos_scheduler_pb2.DeregisterDriverResponse(
                success=False, message="Framework not registered yet."
            )

        if request.id not in self._drivers:
            self._logger.warning(
                "Trying to deregister a driver with id %s, "
                "but no driver with that id is registered.",
                request.id,
            )
            return erdos_scheduler_pb2.DeregisterDriverResponse(
                success=False,
                message=f"Driver with id {request.id} not registered yet.",
            )

        # Deregister the driver.
        driver = self._drivers[request.id]
        completion_time = EventTime(request.timestamp, EventTime.Unit.US)
        self._worker_pool.remove_task(completion_time, driver)
        driver.finish(completion_time)
        del self._drivers[request.id]
        return erdos_scheduler_pb2.DeregisterDriverResponse(
            success=True,
            message=f"Driver with id {request.id} deregistered successfully!",
        )

    async def RegisterTaskGraph(self, request, context):
        """Registers a new TaskGraph with the backend scheduler.
        This is the entry point for a new application of Spark to register
        itself with the backend scheduler, and is intended as an EHLO.
        """
        if not self._initialized:
            self._logger.warning(
                "Trying to register a task graph with ID %s and name %s, "
                "but no framework is registered yet.",
                request.id,
                request.name,
            )
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message="Framework not registered yet.", num_executors=0
            )

        if request.id in self._workload.task_graphs:
            self._logger.warning(
                "The application with ID %s and name %s was already registered.",
                request.id,
                request.name,
            )
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False,
                message=f"Application ID {request.id} with name {request.name} "
                f"already registered!",
                num_executors=0,
            )

        self._logger.info(
            "Attempting to register application ID %s with name %s",
            request.id,
            request.name,
        )
        # Check if query is from TPC-H workload.
        # If yes, retrieve profiled slots and runtime info. If no, use default values
        is_tpch_query = False
        tpch_query_all_stage_info = None
        if request.name.startswith("TPCH Query"):
            is_tpch_query = True
            # retrieve tasks-per-stage and runtime info based on query number
            tpch_query_num = request.name.split("TPCH Query ", 1)[1]
            tpch_query_all_stage_info = get_all_stage_info_for_query(
                tpch_query_num,
                FLAGS.tpch_profile_type,
                FLAGS.tpch_dataset_size,
                FLAGS.tpch_max_executors_per_job)
            same_structure, stage_id_mapping = verify_and_relable_tpch_app_graph(
                query_num=tpch_query_num, dependencies=request.dependencies
            )

            # return failure message if not tpch app isnt of same DAG structure
            if not same_structure:
                self._logger.warning(
                    "TPCH application with ID %s and name %s couldn't be registered."
                    "DAG structure mismatch!",
                    request.id,
                    request.name,
                )
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False,
                    message=f"TPCH application ID {request.id} with name {request.name}"
                    f" couldn't be registered. DAG structure mismatch!",
                    num_executors=0,
                )

        # Construct all the Tasks for the TaskGraph.
        task_ids_to_task: Mapping[int, Task] = {}
        default_resource = Resources(
            resource_vector={Resource(name="Slot_CPU", _id="any"): 20}
        )
        default_runtime = EventTime(
            math.ceil(20 * FLAGS.spark_task_duration_multiplier),
            EventTime.Unit.US
            )
        
        current_time = EventTime(int(time.time()), EventTime.Unit.US)

        for task_dependency in request.dependencies:
            framework_task = task_dependency.key
            if is_tpch_query:
                mapped_stage_id = stage_id_mapping[framework_task.id]
                # Set task_slots to 1 if we are using a profile with fixed
                task_slots = (
                    tpch_query_all_stage_info[mapped_stage_id]["num_tasks"]
                    )
                # Profiled runtime (in ms) * duration_multiplier is converted
                # to nearest second
                task_runtime = math.ceil(
                    (
                        tpch_query_all_stage_info[mapped_stage_id]
                        ["avg_task_duration_ms"]/1000
                        ) * FLAGS.spark_task_duration_multiplier
                        )
                self._logger.info(
                    "Creating Task for given app TPCH stage: %s, mapped to "
                    "original stage id %s, with tasks: %s and avg runtime (s): %s. "
                    "Used multiplier: %s",
                    framework_task.id,
                    mapped_stage_id,
                    task_slots,
                    task_runtime,
                    FLAGS.spark_task_duration_multiplier,
                )
            task_ids_to_task[framework_task.id] = Task(
                name=framework_task.name,
                task_graph=request.id,
                job=Job(
                    name=framework_task.name,
                    profile=WorkProfile(
                        name=f"WorkProfile_{framework_task.name}",
                        execution_strategies=ExecutionStrategies(
                            [
                                ExecutionStrategy(
                                    resources=(
                                        default_resource
                                        if not is_tpch_query
                                        else Resources(
                                            resource_vector={
                                                Resource(
                                                    name="Slot_CPU", _id="any"
                                                ): task_slots
                                            }
                                        )
                                    ),
                                    batch_size=1,
                                    runtime=(
                                        default_runtime
                                        if not is_tpch_query
                                        else EventTime(task_runtime, EventTime.Unit.US)
                                    ),
                                )
                            ]
                        ),
                    ),
                ),
                deadline=EventTime((current_time.time + request.deadline), EventTime.Unit.US),
                # TODO (Sukrit): We should maintain a counter for each application
                # type so that we can correlate the Tasks with a particular invocation.
                timestamp=1,
            )
            # NOTE (Sukrit): We maintain the StageID of the Task as a separate field
            # that is not accessible / used by the Simulator.
            task_ids_to_task[framework_task.id].stage_id = framework_task.id
            self._logger.info(
                "Constructed Task %s for the TaskGraph %s.",
                framework_task.name,
                request.id,
            )

        # Construct the TaskGraph from the Tasks.
        task_graph_structure: Mapping[Task, Sequence[Task]] = {}
        for task_dependency in request.dependencies:
            task_graph_structure[task_ids_to_task[task_dependency.key.id]] = [
                task_ids_to_task[task_id] for task_id in task_dependency.children_ids
            ]
        task_graph = TaskGraph(
            name=request.id,
            tasks=task_graph_structure,
        )
        self._workload.add_task_graph(task_graph)
        self._logger.info(
            "Added the TaskGraph(name=%s, id=%s) to the Workload.",
            request.name,
            request.id,
        )
        self._logger.info(
            "The structure of the TaskGraph %s is \n%s.",
            request.id,
            str(task_graph),
        )

        # Return the response.
        # TODO: (DG) Might want to change the number of initial executors if it causes
        # issues in scaled up expts
        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=f"Application ID {request.id} with name "
            f"{request.name} and deadline {request.deadline} registered successfully!",
            num_executors=FLAGS.initial_executors,
        )

    async def RegisterEnvironmentReady(self, request, context):
        """Registers that the environment (i.e., executors) are ready for the given
        TaskGraph at the specified time.

        This is intended to release the sources of the TaskGraph to the scheduling
        backend, to consider the application in this scheduling cycle.
        """
        if not self._initialized:
            self._logger.warning(
                "Trying to register that the environment is ready for the TaskGraph "
                "with ID %s, but no framework is registered yet.",
                request.id,
            )
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False, message="Framework not registered yet."
            )

        task_graph = self._workload.get_task_graph(request.id)
        if task_graph is None:
            self._logger.warning(
                "Trying to register that the environment is ready for the TaskGraph "
                "with ID %s, but no TaskGraph with that ID is registered.",
                request.id,
            )
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=f"TaskGraph with ID {request.id} not registered yet.",
            )

        if request.num_executors != FLAGS.initial_executors:
            self._logger.warning(
                "The TaskGraph %s requires %s executors, but the environment is ready "
                "with %s executors.",
                request.id,
                FLAGS.initial_executors,
                request.num_executors,
            )
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=f"Number of executors not {FLAGS.initial_executors}.",
            )

        # Release all the sources of the TaskGraph at the given time.
        for source_task in task_graph.get_source_tasks():
            source_task.release(EventTime(request.timestamp, EventTime.Unit.US))

        # Run the scheduler since the Workload has changed.
        await self.run_scheduler()

        return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
            success=True,
            message=f"Environment ready for TaskGraph with ID {request.id}!",
        )

    async def DeregisterFramework(self, request, context):
        """Deregisters the framework with the backend scheduler.
        This is the exit point for a running instance of Spark / Flink to deregister"""
        if not self._initialized:
            self._logger.warning(
                "Trying to deregister the framework at %s, "
                "but no framework is registered yet.",
                request.uri,
            )
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message="Framework not registered yet."
            )

        if not self._master_uri == request.uri:
            self._logger.warning(
                "Trying to deregister the framework at %s, "
                "but the registered framework is at %s.",
                request.uri,
                self._master_uri,
            )
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False,
                message=f"Framework not registered at {request.uri} yet.",
            )

        # Deregister the framework.
        self._initialization_time = None
        self._master_uri = None
        self._initialized = False
        self._logger.info("Deregistering framework at %s", request.uri)
        return erdos_scheduler_pb2.DeregisterFrameworkResponse(
            success=True,
            message=f"Framework at {request.uri} deregistered successfully!",
        )

    async def RegisterWorker(self, request, context):
        """Registers a new worker with the backend scheduler."""
        if not self._initialized:
            self._logger.warning(
                "Trying to register a worker with name %s and id %s, "
                "but no framework is registered yet.",
                request.name,
                request.id,
            )
            return erdos_scheduler_pb2.RegisterWorkerResponse(
                success=False, message="Framework not registered yet."
            )

        # First, we construct the Resources with the given size.
        # TODO (Sukrit): Right now, we drop the memory requirements, we should use
        # them to do multi-dimensional packing using STRL.
        cpu_resource = Resource(name="Slot_CPU")
        # TODO: (DG) Override the request.cores to avoid scaling up physical setup
        worker_resources = Resources(resource_vector={
            cpu_resource: request.cores if not FLAGS.override_worker_cpu_count
            else sys.maxsize}
            )
        self._logger.debug(
            "Successfully constructed the resources for the worker %s: %s.",
            request.name,
            worker_resources,
        )

        # Construct a new Worker instance, and add it to the WorkerPool.
        worker = Worker(
            name=request.id,
            resources=worker_resources,
        )
        self._worker_pool.add_workers([worker])

        self._logger.info(
            "Registering worker with name %s, and resources %s.",
            worker.name,
            worker_resources,
        )

        # Run the scheduler since the Resource set has changed, and new task graphs
        # may become eligible to run.
        await self.run_scheduler()

        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True,
            message=f"Worker {request.name} registered successfully!",
            cores=FLAGS.virtualized_cores,
            memory=FLAGS.virtualized_memory * 1024,
        )

    async def NotifyTaskCompletion(self, request, context):
        """Notifies the backend scheduler that a task has completed."""
        if not self._initialized:
            self._logger.warning(
                "Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, "
                "but no framework is registered yet.",
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False, message="Framework not registered yet."
            )

        task_graph = self._workload.get_task_graph(request.application_id)
        if task_graph is None:
            self._logger.warning(
                "Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, but the application "
                "was not registered with the backend yet.",
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=f"Application with ID {request.application_id} "
                f"not registered yet.",
            )

        # Find the Task that has completed, and mark it as such.
        matched_task = None
        for task in task_graph.get_nodes():
            if task.stage_id == request.task_id:
                matched_task = task
        if matched_task is None:
            self._logger.warning(
                "Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, but the task "
                "was not found in the TaskGraph.",
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=f"Task with ID {request.task_id} "
                f"not found in TaskGraph {request.application_id}.",
            )

        # Instead of completing & removing the task immediately, check
        # if it is actually complete or will complete in the future

        # Get the actual task completion timestamp
        actual_task_completion_time = (
            matched_task.start_time.time + matched_task.remaining_time.time
        )

        current_time = EventTime(int(time.time()), EventTime.Unit.US)
        self._logger.info(
            "Received task for completion at time: %s , task.start_time: %s ,"
            "task.remaining_time (=runtime):  %s ,  actual completion time: %s. "
            "Task details: %s",
            current_time.time,
            matched_task.start_time.time,
            matched_task.remaining_time.time,
            actual_task_completion_time,
            matched_task,
        )

        # TODO DG: remaining_time assumes execution of the slowest strategy
        # Should be updated to reflect correct remaining_time based on chosen strategy?

        # Add all tasks to _tasks_marked_for_completion queue.
        # If task has actually completed, it will be dequeued immediately
        # Else it will be dequeued at its actual task completion time
        self._tasks_marked_for_completion.put(
            TimedItem(actual_task_completion_time, matched_task)
        )

        # NOTE: task.finish() and run_scheduler() invocations are postponed
        # until it is time for the task to be actually marked as complete.

        return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
            success=True,
            message=f"Task with ID {request.task_id} marked for completion at "
            f"{current_time}! It will be removed on actual "
            f"task completion time at {actual_task_completion_time}",
        )

    async def GetPlacements(self, request, context):
        """Retrieves the placements applicable at the specified time."""
        request_timestamp = EventTime(request.timestamp, EventTime.Unit.US)
        if not self._initialized:
            self._logger.warning(
                "Trying to get placements for %s at time %s, "
                "but no framework is registered yet.",
                request.id,
                request_timestamp,
            )
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False, message="Framework not registered yet."
            )

        if request.id not in self._placements:
            self._logger.warning(
                "Trying to get placements for %s at time %s, but the application "
                "was not registered with the backend yet.",
                request.id,
                request_timestamp,
            )

        # Construct and return the placements.,
        placements = []
        clip_at = -1
        for index, placement in enumerate(self._placements[request.id]):
            if placement.placement_time <= request_timestamp:
                clip_at = index
                # TODO: (DG) Due to small dataset size, each stage automatically gets
                # one data partition i.e. one task and one executor. But later for
                # large datasets, we might leverage use_profile_to_scale_executors
                self._logger.info(
                    f"Going to set placement.task to run: {placement}"
                )

                # Mark the Task as RUNNING.
                placement.task.start(request_timestamp)

                # resources = placement.execution_strategy.resources
                placements.append(
                    erdos_scheduler_pb2.Placement(
                        worker_id=placement.worker_id,
                        application_id=request.id,
                        task_id=placement.task.stage_id,
                        cores=1,
                    )
                )
        self._placements[request.id] = self._placements[request.id][clip_at + 1 :]
        self._logger.info(
            "Constructed %s placements at time %s for application with ID %s.",
            len(placements),
            request.timestamp,
            request.id,
        )
        return erdos_scheduler_pb2.GetPlacementsResponse(
            success=True,
            placements=placements,
            message=f"Constructed {len(placements)} "
            f"placements at time {request.timestamp}.",
        )

    # Function to pop tasks from queue based on actual completion time
    async def PopTasksBasedOnTime(self):
        while True:
            if not self._tasks_marked_for_completion.empty():
                # Get the top item from the priority queue
                top_item = self._tasks_marked_for_completion._queue[0][1]

                # Check if top item's timestamp is reached or passed by current time
                current_time = EventTime(int(time.time()), EventTime.Unit.US)
                if top_item.timestamp <= current_time.time:
                    # Pop the top item
                    popped_item = self._tasks_marked_for_completion.get()
                    self._logger.info(
                        "Removing task from pending completion queue at time: %s. "
                        "Task details: %s",
                        current_time,
                        popped_item.task,
                    )

                    # Mark the Task as completed.
                    # Also release the task from the scheduler service
                    popped_item.task.update_remaining_time(EventTime.zero())
                    popped_item.task.finish(current_time)

                    # Run the scheduler since the Workload has changed.
                    await self.run_scheduler()
                
                else:
                    # If the top item's timestamp hasn't been reached yet,
                    # sleep for a short duration
                    await asyncio.sleep(0.1)  # TODO: Can adjust value, curr=0.1s
            else:
                # If the queue is empty, sleep for a short duration
                await asyncio.sleep(0.1)  # TODO: Can adjust value, curr=0.1s


async def serve():
    """Serves the ERDOS Scheduling RPC Server."""
    # Initialize the server.
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=FLAGS.max_workers))
    erdos_scheduler_pb2_grpc.add_SchedulerServiceServicer_to_server(
        SchedulerServiceServicer(), server
    )

    # Start the server.
    server.add_insecure_port(f"[::]:{FLAGS.port}")
    await server.start()
    print("Initialized ERDOS Scheduling RPC Server on port", FLAGS.port)
    await server.wait_for_termination()


def main(argv):
    # Parse the command-line flags
    flags.FLAGS(argv)

    # Access the value of the flag
    multiplier = flags.FLAGS.spark_task_duration_multiplier
    override_worker_cpus = flags.FLAGS.override_worker_cpu_count
    
    # Your application logic here
    print("Multiplier:", multiplier)
    print("Override worker CPUs:", override_worker_cpus)
    
    # Create an asyncio event loop
    loop = asyncio.get_event_loop()

    # Run the event loop until serve() completes
    try:
        loop.run_until_complete(serve())
    finally:
        loop.close()


if __name__ == "__main__":
    app.run(main)
