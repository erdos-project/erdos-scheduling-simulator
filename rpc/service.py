import asyncio
import heapq
import math
import os
import sys
import time
from concurrent import futures
from operator import attrgetter
from typing import Dict, Mapping, Sequence
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
    TaskState,
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
    "tpch_dataset_size", "50", ["50", "100", "250", "500"], "Options in GB eg. 50g for "
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
    4,
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
    6,
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
flags.DEFINE_float(
    "task_graph_slo_factor",
    2.0,
    "The multiplicative factor to be used with critical path length of the task graph. "
    "This helps inform the deadline for the taskgraph and all tasks within the task "
    "graph. The value be > 1.0 since the taskgraph would take atleast the critical path "
    "time duration to complete.",
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
        self._logger = setup_logging(
            name=__name__,
            log_dir=FLAGS.log_dir,
            log_file=FLAGS.log_file_name,
            log_level=FLAGS.log_level
        )
        self._initialized = False
        self._initialization_time = -1
        self._last_step_up_time = EventTime.zero()
        self._master_uri = None

        # The simulator types maintained by the Servicer.
        self._worker_pool = None
        self._worker_pools = None
        self._drivers: Mapping[str, Task] = {}
        self._workload = None

        # Track taskgraph completion progress.
        self._total_taskgraphs_registered = 0
        self._total_taskgraphs_missed = 0
        self._total_taskgraphs_met = 0
        self._total_taskgraphs_cancelled = 0
        self._cancelled_taskgraphs = set()
        self._task_graph_slo_factor = FLAGS.task_graph_slo_factor

        # Scheduler information maintained by the servicer.
        self._scheduler_running_lock = asyncio.Lock()
        self._scheduler_running = False
        self._rerun_scheduler = False
        self._scheduler_is_task_type = False
        if FLAGS.scheduler == "EDF":
            self._scheduler = EDFScheduler(
                enforce_deadlines=FLAGS.enforce_deadlines,
                _flags=FLAGS,
            )
            self._scheduler_is_task_type = True
        elif FLAGS.scheduler == "FIFO":
            self._scheduler = FIFOScheduler(
                enforce_deadlines=FLAGS.enforce_deadlines,
                _flags=FLAGS,
            )
            self._scheduler_is_task_type = True
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
                log_to_file=True,
            )
            self._scheduler_is_task_type = not FLAGS.release_taskgraphs
        else:
            raise ValueError(f"Unknown scheduler {FLAGS.scheduler}.")

        # Placement information maintained by the servicer.
        # The placements map the application IDs to the Placement retrieved from the
        # scheduler.
        # NOTE: (DG) This is a new nested dict implementation.
        # First level of dict is a mapping from app-id to all tasks in that app-id
        # Second level of dict is a mapping from tasks to exact placement.
        # TODO: (DG) This will no longer be ordered by time, so the check needs to be
        # done for all tasks? Also, we might need to delete the placement once executed?
        self._placements: Dict[str, Dict[str, Placement]] = {}

        # _executed_placements keep a track of previously completed placements since
        # placements are deleted after being released. Can be used for debugging.
        self._executed_placements: Dict[str, Placement] = {}

        # Additional task information maintained by the servicer
        self._tasks_marked_for_completion = PriorityQueue()

        # Start the asyncio loop for clearing out pending tasks for completion
        asyncio.create_task(self.PopTasksBasedOnTime())

        super().__init__()

    async def schedule(self) -> None:
        """Schedules the tasks that have been added to the Workload."""
        current_time = EventTime(int(time.time()), EventTime.Unit.US) - self._initialization_time

        async with self._scheduler_running_lock:
            if self._scheduler_running:
                self._logger.error(
                    "[%s] Scheduler already running, this should never be reached.",
                    current_time,
                )
                return
            self._scheduler_running = True

        self._logger.info(
            "[%s] Starting a scheduling cycle with %s TaskGraphs and %s Workers.",
            current_time,
            len(self._workload.task_graphs),
            len(self._worker_pool.workers),
        )

        # Cumulate the resources from all the WorkerPools
        for worker_pool in self._worker_pools.worker_pools:
            worker_pool_resources = worker_pool.resources
            for resource_name in set(
                map(lambda value: value[0].name, worker_pool_resources.resources)
            ):
                resource = Resource(name=resource_name, _id="any")
                self._logger.info(
                    f"{current_time},WORKER_POOL_UTILIZATION,{worker_pool.id},"
                    f"{resource_name},"
                    f"{worker_pool_resources.get_allocated_quantity(resource)},"
                    f"{worker_pool_resources.get_available_quantity(resource)}"
                )
        
        # Perform worker pool step
        self._logger.info(
            "[%s] Need to perform a step before schedule().",
            current_time,
            )
        completed_tasks = self.PerformWorkerPoolStep(sim_time=current_time)
        
        # Finish all tasks that have now completed
        for completed_task in completed_tasks:
            self.CleanupTaskExecution(
                task=completed_task,
                sim_time=current_time
                )
        

        # TODO (Sukrit): Change this to a better implementation.
        # Let's do some simple scheduling for now, that gives a fixed number of
        # executors to all the available applications in intervals of 10 seconds.
        if len(self._workload.task_graphs) >= 1:
            scheduler_placements = self._scheduler.schedule(
                sim_time=EventTime(current_time.time, EventTime.Unit.US),
                workload=self._workload,
                worker_pools=self._worker_pools,
            )
            
            # Filter the scheduler_placements that are now in CANCEL_TASK state.
            cancel_task_placements = list(filter(
                lambda p: p.placement_type == Placement.PlacementType.CANCEL_TASK,
                scheduler_placements,
            ))
            self._logger.info(
                "[%s] Received %s tasks to be cancelled: %s.",
                current_time,
                len(cancel_task_placements),
                cancel_task_placements,
            )
            # Issue task cancellations for identified tasks and taskgraphs so that
            # the taskgraphs are no longer in consideration
            for placement in cancel_task_placements:
                # Update the task placement decision so that we can stop
                # responding to RPC calls from its driver based on CANCEL_TASK type

                if placement.task.task_graph not in self._placements:
                    self._placements[placement.task.task_graph] = {}
                    self._logger.warning(
                        "[%s] Came to cancel a placement but taskgraph %s was not in "
                        "self._placements. Creating an empty dict entry.",
                        current_time,
                        placement.task.task_graph,
                    )
                self._placements[placement.task.task_graph][placement.task] = placement
                self._logger.info(
                        "[%s] Added cancel placement to taskgraph %s for task %s. "
                        "Placement: %s",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                        placement,
                    )

                # Since even one task getting cancelled, implies task-graph
                # cancellation, we add the task-graph to cancelled set
                if placement.task.task_graph not in self._cancelled_taskgraphs:
                    self._cancelled_taskgraphs.add(placement.task.task_graph)
                    self._total_taskgraphs_cancelled += 1
                    self._logger.info(
                            "[%s] RUN_STATS (registered, met, missed, cancelled): %s, %s, %s, %s",
                            current_time,
                            self._total_taskgraphs_registered,
                            self._total_taskgraphs_met,
                            self._total_taskgraphs_missed,
                            self._total_taskgraphs_cancelled,
                            )
                
                self._logger.info(
                    "[%s] Cancelling task: %s from taskgraph: %s",
                    current_time,
                    placement.task.name,
                    placement.task.task_graph,
                )
                # Sending tasks to cancel.
                placement.task.cancel(
                    time=current_time,
                )

                # Also cancel the task-graph so that all dependent tasks are removed
                task_graph = self._workload.get_task_graph(placement.task.task_graph)
                if task_graph is None:
                    self._logger.error("[%s] No TaskGraph found for %s",
                                       current_time,
                                       placement.task.task_graph,
                                       )
                
                for cancelled_task in task_graph.cancel(placement.task, current_time):
                    self._logger.info(
                        "[%s] Further cancelling dependent task: %s from taskgraph: %s",
                        current_time,
                        placement.task.name,
                        placement.task.task_graph,
                    )
                    
                    cancelled_task.cancel(
                        time=current_time,
                    )

                # TODO: (DG): Ensure that task-graph is removed from the workload and
                # doesn't show up in the next iteration of tetrisched scheduler?
            
            # Filter the scheduler_placements that are not of type PLACE_TASK and 
            # have not been placed.
            filtered_placements = filter(
                lambda p: p.placement_type == Placement.PlacementType.PLACE_TASK
                and p.is_placed(),
                scheduler_placements,
            )
            for placement in sorted(
                filtered_placements, key=attrgetter("placement_time")
            ):
                if placement.task.task_graph not in self._placements:
                    self._placements[placement.task.task_graph] = {}
                    self._logger.info(
                        "[%s] Want to add a placement but taskgraph %s was not in "
                        "self._placements. Creating an empty dict entry.",
                        current_time,
                        placement.task.task_graph,
                    )
                if placement.task not in self._placements[placement.task.task_graph]:
                    self._logger.info(
                        "[%s] Adding new placement to taskgraph %s for task %s. "
                        "Placement: %s",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                        placement,
                    )
                else:
                    self._logger.info(
                        "[%s] Updating an existing placement in taskgraph %s for task %s. "
                        "Placement: %s",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                        placement,
                    )
                self._placements[placement.task.task_graph][placement.task] = placement
                
                # Schedule the task here since marking it as running requires it to be
                # scheduled before. We mark it to be running when we inform the
                # framework of the placement.

                # TODO: (DG) ASK - dont think tasks need to be marked as unscheduled on cancellation?
                placement.task.schedule(
                    time=placement.placement_time,
                    placement=placement,
                )

            
            # Handle task placements that have returned with unplaced tasks
            unplaced_placements = filter(
                lambda p: p.placement_type == Placement.PlacementType.PLACE_TASK
                and not p.is_placed(),
                scheduler_placements,
            )
            for placement in unplaced_placements:
                if placement.task.task_graph not in self._placements:
                    self._logger.info(
                        "[%s] taskgraph %s not found for task %s, couldn't invalidate it or"
                        "it was previously invalidated.",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                    )
                elif placement.task in self._placements[placement.task.task_graph]:
                    self._logger.info(
                        "[%s] Invalidated the placement (taskgraph %s and task %s)"
                        "from self._placements along with entire taskgraph.",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                    )
                    for task in self._placements[placement.task.task_graph]:
                        self._logger.info(
                            "[%s] Invalidating the placement for task %s "
                            "from self._placements due to invalidation of %s.",
                            current_time,
                            task,
                            placement.task,
                        )
                        # Unschedule the task
                        if task.state is TaskState.SCHEDULED:
                            task.unschedule(time=current_time)
                        else:
                            self._logger.warning(
                                "[%s] Could not unschedule since task %s was "
                                "found in state %s in during invalidation of %s.",
                                current_time,
                                task,
                                task.state,
                                placement.task,
                            )
                    # delete the taskgraph at once since we cant change size
                    # of dict while iterating
                    del self._placements[placement.task.task_graph]
                else:
                    self._logger.info(
                        "[%s] Couldn't invalidate placement (taskgraph %s and task %s)."
                        "It couldnt be found in self._placements.",
                        current_time,
                        placement.task.task_graph,
                        placement.task,
                    )

        scheduler_end_time = EventTime(int(time.time()), EventTime.Unit.US) - self._initialization_time
        self._logger.info(
            "[%s] Finished the scheduling cycle initiated at %s.",
            scheduler_end_time,
            current_time,
        )

        # Check if another run of the Scheduler has been requested, and if so, create
        # a task for it. Otherwise, mark the scheduler as not running.
        async with self._scheduler_running_lock:
            self._scheduler_running = False
            self._logger.info("[%s] self._rerun_scheduler: %s.",
                              scheduler_end_time,
                              self._rerun_scheduler,
                              )
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
                "Framework already registered at %s with the address %s at %s",
                self._initialization_time,
                self._master_uri,
                self._initialization_time,
            )
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=f"Framework already registered at "
                f"{self._initialization_time} at the address {self._master_uri}",
            )

        # Setup a new Framework instance.
        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = EventTime(request.timestamp, EventTime.Unit.US)
        self._initialized = True
        self._logger.info(
            "[%s] Registering framework %s with URI %s.",
            self._initialization_time,
            framework_name,
            self._master_uri,
        )

        # Setup the simulator types.
        parsed_uri = urlparse(self._master_uri)
        self._worker_pool = WorkerPool(
            name=f"WorkerPool_{parsed_uri.netloc}",
            _logger=self._logger
            )
        self._worker_pools = WorkerPools(worker_pools=[self._worker_pool])
        self._workload = Workload.from_task_graphs({})

        # Return the response.
        return erdos_scheduler_pb2.RegisterFrameworkResponse(
            success=True,
            message=f"{framework_name} at {self._master_uri} registered successfully!",
        )

    async def RegisterDriver(self, request, context):
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
        
        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to register a driver with name %s and id %s, "
                "but no framework is registered yet.",
                sim_time,
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
            "[%s] Received a request to register a driver with name %s, URI: %s. "
            "The driver requires %s cores and %s memory.",
            sim_time,
            request.id,
            request.uri,
            request.cores,
            request.memory,
        )
        driver_resources = Resources(
            resource_vector={Resource(name="Slot_CPU", _id="any"): 1},
            _logger=self._logger,
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
            _logger=self._logger,
        )
        self._drivers[request.id] = driver

        # Iterate over the Workers and find a Worker that can accomodate the driver.
        placement_found = False
        for worker in self._worker_pool.workers:
            for execution_strategy in driver.available_execution_strategies:
                if worker.can_accomodate_strategy(execution_strategy):
                    # This Worker can accomodate the Driver, we assign it here.
                    placement_found = True
                    # self._worker_pool.place_task(driver, execution_strategy, worker.id)

                    # Update the Task's state and placement information.
                    placement_time = sim_time
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
                        message=f"[{sim_time}] Driver {request.id} registered successfully!",
                        worker_id=worker.name,
                    )

        if not placement_found:
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message=f"[{sim_time}] No Worker can accomodate the driver {request.id} yet.",
                worker_id="",
            )

    async def DeregisterDriver(self, request, context):
        completion_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
        
        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to deregister a driver with id %s, "
                "but no framework is registered yet.",
                completion_time,
                request.id,
            )
            return erdos_scheduler_pb2.DeregisterDriverResponse(
                success=False, message="Framework not registered yet."
            )

        if request.id not in self._drivers:
            self._logger.warning(
                "[%s] Trying to deregister a driver with id %s, "
                "but no driver with that id is registered.",
                completion_time,
                request.id,
            )
            return erdos_scheduler_pb2.DeregisterDriverResponse(
                success=False,
                message=f"[{completion_time}] Driver with id {request.id} not registered yet.",
            )

        # Deregister the driver.
        driver = self._drivers[request.id]
        # self._worker_pool.remove_task(completion_time, driver)
        driver.finish(completion_time)
        del self._drivers[request.id]
        return erdos_scheduler_pb2.DeregisterDriverResponse(
            success=True,
            message=f"[{completion_time}] Driver with id {request.id} deregistered successfully!",
        )

    async def RegisterTaskGraph(self, request, context):
        """Registers a new TaskGraph with the backend scheduler.
        This is the entry point for a new application of Spark to register
        itself with the backend scheduler, and is intended as an EHLO.
        """
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time

        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to register a task graph with ID %s and name %s, "
                "but no framework is registered yet.",
                sim_time,
                request.id,
                request.name,
            )
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message="Framework not registered yet.", num_executors=0
            )

        if request.id in self._workload.task_graphs:
            self._logger.warning(
                "[%s] The application with ID %s and name %s was already registered.",
                sim_time,
                request.id,
                request.name,
            )
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False,
                message=f"[{sim_time}] Application ID {request.id} with name "
                f"{request.name} already registered!",
                num_executors=0,
            )

        self._logger.info(
            "[%s] Attempting to register application ID %s with name %s",
            sim_time,
            request.id,
            request.name,
        )
        # Check if query is from TPC-H workload.
        # If yes, retrieve profiled slots and runtime info. If no, use default values
        is_tpch_query = False
        tpch_query_all_stage_info = None
        if request.name.startswith("TPCH Query"):
            is_tpch_query = True
            # retrieve tasks-per-stage and runtime info based on query specifications
            # Split the string by spaces
            query_parts = request.name.split()
            
            # Initialize dataset_size and max_executor variables with default
            tpch_query_num = None
            tpch_dataset_size = int(FLAGS.tpch_dataset_size)
            tpch_max_executors_per_job = int(FLAGS.tpch_max_executors_per_job)
            
            # Check if the string has the required format
            # Format 1: "TPCH Query <queryNum>"
            # Format 2: "TPCH Query <queryNum> <datasetSize> <maxExecutors>"
            if len(query_parts) >= 3 and query_parts[0] == "TPCH" and query_parts[1] == "Query":
                tpch_query_num = int(query_parts[2])
                
                # If dataset size and max cores are provided
                if len(query_parts) >= 5:
                    tpch_dataset_size = int(query_parts[3])
                    tpch_max_executors_per_job = int(query_parts[4])

            tpch_query_all_stage_info = get_all_stage_info_for_query(
                query_num=tpch_query_num,
                profile_type=FLAGS.tpch_profile_type,
                dataset_size=tpch_dataset_size,
                max_executors=tpch_max_executors_per_job)
            
            same_structure, stage_id_mapping = verify_and_relable_tpch_app_graph(
                query_num=tpch_query_num, dependencies=request.dependencies
            )

            # return failure message if not tpch app isnt of same DAG structure
            if not same_structure:
                self._logger.warning(
                    "[%s] TPCH application with ID %s and name %s couldn't be registered."
                    "DAG structure mismatch!",
                    sim_time,
                    request.id,
                    request.name,
                )
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False,
                    message=f"[{sim_time}] TPCH application ID {request.id} with name {request.name}"
                    f" couldn't be registered. DAG structure mismatch!",
                    num_executors=0,
                )

        # Construct all the Tasks for the TaskGraph.
        task_ids_to_task: Mapping[int, Task] = {}
        default_resource = Resources(
            resource_vector={Resource(name="Slot_CPU", _id="any"): 20},
            _logger=self._logger,
        )
        default_runtime = EventTime(
            math.ceil(20 * FLAGS.spark_task_duration_multiplier),
            EventTime.Unit.US
            )

        for i, task_dependency in enumerate(request.dependencies):
            framework_task = task_dependency.key
            if is_tpch_query:
                mapped_stage_id = stage_id_mapping[framework_task.id]
                
                # NOTE: task_slots and task_runtime given to scheduler might be updated
                # based on tpch_max_executors_per_job. If task_slots >
                # tpch_max_executors_per_job, we transform (task_slots * task_runtime)
                # as tpch_max_executors_per_job * (
                #           (task_slots * task_runtime)/tpch_max_executors_per_job
                # )
                # TODO: (DG) It is not foolproof since scheduler can give more than
                # tpch_max_executors_per_job to app if it decides to run multiple
                # independent stages in parallel

                profiled_task_slots = (
                    tpch_query_all_stage_info[mapped_stage_id]["num_tasks"]
                    )
                # Profiled runtime (in ms) * duration_multiplier is converted
                # to nearest second
                profiled_task_runtime = math.ceil(
                    (
                        tpch_query_all_stage_info[mapped_stage_id]
                        ["avg_task_duration_ms"]/1000
                        ) * FLAGS.spark_task_duration_multiplier
                        )
                
                task_slots = (profiled_task_slots
                              if profiled_task_slots <= tpch_max_executors_per_job
                              else tpch_max_executors_per_job
                              )
                # Setting minimum task_runtime to 3s to allow stages to complete
                task_runtime = max(3, (
                    profiled_task_runtime
                    if profiled_task_slots <= tpch_max_executors_per_job
                    else math.ceil(
                        (profiled_task_slots *
                         profiled_task_runtime)/tpch_max_executors_per_job)
                    )
                )
                if profiled_task_slots > tpch_max_executors_per_job:
                    self._logger.info(
                        "[%s] Profiled slots > tpch_max_executors_per_job: %s. Converted "
                        "(slots,runtime) from (%s,%s) to (%s, %s)",
                        sim_time,
                        tpch_max_executors_per_job,
                        profiled_task_slots,
                        profiled_task_runtime,
                        task_slots,
                        task_runtime,
                        )
                    
                self._logger.info(
                    "[%s] Creating Task for given app TPCH stage: %s, mapped to "
                    "original stage id %s, with tasks: %s and avg runtime (s): %s. "
                    "Used multiplier: %s",
                    sim_time,
                    framework_task.id,
                    mapped_stage_id,
                    task_slots,
                    task_runtime,
                    FLAGS.spark_task_duration_multiplier,
                )
            task_ids_to_task[framework_task.id] = Task(
                name=f"task_{framework_task.name}_{i}",
                task_graph=request.id,
                job=Job(
                    name=f"job_{framework_task.name}_{i}",
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
                                            },
                                            _logger=self._logger,
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
                # NOTE: (DG) Removed setting deadline here and will set deadline 
                # based on taskgraphs critical path instead.
                deadline=EventTime.invalid(),
                
                # TODO (Sukrit): We should maintain a counter for each application
                # type so that we can correlate the Tasks with a particular invocation.
                timestamp=1,
                _logger=self._logger,
            )
            # NOTE (Sukrit): We maintain the StageID of the Task as a separate field
            # that is not accessible / used by the Simulator.
            task_ids_to_task[framework_task.id].stage_id = framework_task.id
            self._logger.info(
                "[%s] Constructed Task %s for the TaskGraph %s.",
                sim_time,
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

        # Calculating critical path time from task graph
        critical_path = task_graph.get_longest_path(
                weights=lambda task: (task.slowest_execution_strategy.runtime.time)
            )
        critical_path_time = (
            sum(
                [t.slowest_execution_strategy.runtime for t in critical_path],
                start=EventTime.zero(),
            )
            .to(EventTime.Unit.US)
            .time
        )

        # Setting taskgraph and task deadlines using critical_path_time * SLO factor
        task_graph_slo_time = math.ceil(
            critical_path_time * self._task_graph_slo_factor
            )
        
        for task in task_graph.get_nodes():
            deadline = EventTime(sim_time.time + task_graph_slo_time,
                                 unit=EventTime.Unit.US
                                 )
            task.update_deadline(deadline)

        task_graph.to_dot(f"{request.id}.dot")
        self._workload.add_task_graph(task_graph)
        self._logger.info(
            "[%s] Added the TaskGraph(name=%s, id=%s, deadline=%s, critical_path_time = %s, task_graph_slo_time = %s) to the Workload.",
            sim_time,
            request.name,
            request.id,
            task_graph.deadline,
            critical_path_time,
            task_graph_slo_time,
        )
        self._logger.info(
            "[%s] The structure of the TaskGraph %s is \n%s.",
            sim_time,
            request.id,
            str(task_graph),
        )

        # Increment total number of taskgraphs registered.
        self._total_taskgraphs_registered += 1

        # Show current run statistics.
        self._logger.info(
            "[%s] RUN_STATS (registered, met, missed, cancelled): %s, %s, %s, %s",
            sim_time,
            self._total_taskgraphs_registered,
            self._total_taskgraphs_met,
            self._total_taskgraphs_missed,
            self._total_taskgraphs_cancelled,
            )
        
        # Return the response.
        # TODO: (DG) Might want to change the number of initial executors if it causes
        # issues in scaled up expts
        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=f"[{sim_time}] Application ID {request.id} with name "
            f"{request.name} and deadline {task_graph.deadline} registered successfully!",
            num_executors=FLAGS.initial_executors,
        )

    async def RegisterEnvironmentReady(self, request, context):
        """Registers that the environment (i.e., executors) are ready for the given
        TaskGraph at the specified time.

        This is intended to release the sources of the TaskGraph to the scheduling
        backend, to consider the application in this scheduling cycle.
        """
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time

        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to register that the environment is ready for the TaskGraph "
                "with ID %s, but no framework is registered yet.",
                sim_time,
                request.id,
            )
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False, message=f"[{sim_time}] Framework not registered yet."
            )
        
        task_graph = self._workload.get_task_graph(request.id)
        if task_graph is None:
            self._logger.warning(
                "[%s] Trying to register that the environment is ready for the TaskGraph "
                "with ID %s, but no TaskGraph with that ID is registered.",
                sim_time,
                request.id,
            )
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=f"[{sim_time}] TaskGraph with ID {request.id} not registered yet.",
            )

        if request.num_executors != FLAGS.initial_executors:
            self._logger.warning(
                "[%s] The TaskGraph %s requires %s executors, but the environment is ready "
                "with %s executors.",
                sim_time,
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
            source_task.release(sim_time)

        self._logger.info(f"[{sim_time}] Environment ready for TaskGraph with ID {request.id}!")

        # Run the scheduler since the Workload has changed.
        await self.run_scheduler()

        return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
            success=True,
            message=f"[{sim_time}] Environment ready for TaskGraph with ID {request.id}!",
        )

    async def DeregisterFramework(self, request, context):
        """Deregisters the framework with the backend scheduler.
        This is the exit point for a running instance of Spark / Flink to deregister"""
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
        
        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to deregister the framework at %s, "
                "but no framework is registered yet.",
                sim_time,
                request.uri,
            )
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message=f"[{sim_time}] Framework not registered yet."
            )

        if not self._master_uri == request.uri:
            self._logger.warning(
                "[%s] Trying to deregister the framework at %s, "
                "but the registered framework is at %s.",
                sim_time,
                request.uri,
                self._master_uri,
            )
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False,
                message=f"[{sim_time}] Framework not registered at {request.uri} yet.",
            )

        # Deregister the framework.
        self._initialization_time = None
        self._master_uri = None
        self._initialized = False
        self._logger.info("[%s] Deregistering framework at %s", sim_time, request.uri)
        return erdos_scheduler_pb2.DeregisterFrameworkResponse(
            success=True,
            message=f"[{sim_time}] Framework at {request.uri} deregistered successfully!",
        )

    async def RegisterWorker(self, request, context):
        """Registers a new worker with the backend scheduler."""
        current_time = EventTime(int(time.time()), EventTime.Unit.US) - self._initialization_time
        
        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to register a worker with name %s and id %s, "
                "but no framework is registered yet.",
                current_time,
                request.name,
                request.id,
            )
            return erdos_scheduler_pb2.RegisterWorkerResponse(
                success=False, message=f"[{current_time}] Framework not registered yet."
            )

        # First, we construct the Resources with the given size.
        # TODO (Sukrit): Right now, we drop the memory requirements, we should use
        # them to do multi-dimensional packing using STRL.
        cpu_resource = Resource(name="Slot_CPU")
        # TODO: (DG) Override the request.cores to avoid scaling up physical setup
        worker_resources = Resources(
            resource_vector={
                cpu_resource: request.cores if not FLAGS.override_worker_cpu_count
                else 640
                },
            _logger=self._logger,
            )
        self._logger.debug(
            "[%s] Successfully constructed the resources for the worker %s: %s.",
            current_time,
            request.name,
            worker_resources,
        )

        # Construct a new Worker instance, and add it to the WorkerPool.
        worker = Worker(
            name=request.id,
            resources=worker_resources,
            _logger=self._logger,
        )
        self._worker_pool.add_workers([worker])

        self._logger.info(
            "[%s] Registering worker with name %s, and resources %s.",
            current_time,
            worker.name,
            worker_resources,
        )

        # Run the scheduler since the Resource set has changed, and new task graphs
        # may become eligible to run.
        await self.run_scheduler()

        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True,
            message=f"[{current_time}] Worker {request.name} registered successfully!",
            cores=FLAGS.virtualized_cores,
            memory=FLAGS.virtualized_memory * 1024,
        )

    async def NotifyTaskCompletion(self, request, context):
        """Notifies the backend scheduler that a task has completed."""
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
        
        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, "
                "but no framework is registered yet.",
                sim_time,
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False, message="Framework not registered yet."
            )

        task_graph = self._workload.get_task_graph(request.application_id)
        if task_graph is None:
            self._logger.warning(
                "[%s] Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, but the application "
                "was not registered with the backend yet.",
                sim_time,
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=f"[{sim_time}] Application with ID {request.application_id} "
                f"not registered yet.",
            )

        # Find the Task that has completed, and mark it as such.
        matched_task = None
        for task in task_graph.get_nodes():
            if task.stage_id == request.task_id:
                matched_task = task
        if matched_task is None:
            self._logger.warning(
                "[%s] Trying to notify the backend scheduler that the task with ID %s "
                "from application %s has completed, but the task "
                "was not found in the TaskGraph.",
                sim_time,
                request.task_id,
                request.application_id,
            )
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=f"[{sim_time}] Task with ID {request.task_id} "
                f"not found in TaskGraph {request.application_id}.",
            )

        # Instead of completing & removing the task immediately, check
        # if it is actually complete or will complete in the future

        # Get the actual task completion timestamp
        actual_task_completion_time = (
            matched_task.start_time.time + matched_task.remaining_time.time
        )

        self._logger.info(
            "[%s] Received task for completion. task.start_time: %s ,"
            "task.remaining_time:  %s ,  actual completion time: %s. "
            "Task details: %s",
            sim_time.time,
            matched_task.start_time.time,
            matched_task.remaining_time.time,
            actual_task_completion_time,
            matched_task,
        )

        if sim_time.time > actual_task_completion_time:
            self._logger.warning(
                "[%s] Task exceeded actual completion time by %s, "
                "Task details: %s",
                sim_time.time,
                (sim_time.time - actual_task_completion_time),
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
            f"{sim_time}! It will be removed on actual "
            f"task completion time at {actual_task_completion_time}",
        )

    async def GetPlacements(self, request, context):
        """Retrieves the placements applicable at the specified time."""
        sim_time = EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time

        if not self._initialized:
            self._logger.warning(
                "[%s] Trying to get placements for %s, "
                "but no framework is registered yet.",
                sim_time,
                request.id,
            )
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False, message="Framework not registered yet."
            )

        if request.id not in self._placements:
            self._logger.warning(
                "[%s] Trying to get placements for %s, but the application "
                "was not registered with the backend yet OR was cancelled.",
                sim_time,
                request.id,
            )
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False, 
                message=f"[{sim_time}] Trying to get placements for "
                f"{request.id}, but the application was not registered with the "
                f"backend yet OR was cancelled."
            )
        
        # Construct and return the placements.,
        placements = []
        
        # Keep track of app_ids and task_names to delete after placements are issued
        to_delete = []
        
        for task in self._placements[request.id].keys():
            task_placement = self._placements[request.id][task]
            if task.state is TaskState.CANCELLED:
                # Task cancelled, add to list to remove from self._placements
                to_delete.append((request.id, task))
            else:
                if task_placement.placement_time <= sim_time:
                    # TODO: (DG) Due to small dataset size, each stage automatically gets
                    # one data partition i.e. one task and one executor. But later for
                    # large datasets, we might leverage use_profile_to_scale_executors
                    # to modify the placement before it is sent
                    self._logger.info(
                        f"[{sim_time}] Going to set placement.task to run: {task_placement}"
                    )

                    # Mark the Task as RUNNING.
                    # Right now we don't run task.start() if
                    # task is already in RUNNING or CANCELLED state.
                    # Only SCHEDULED -> RUNNING transition is allowed.
                    if task.state == TaskState.SCHEDULED:
                        try:
                            # Initialize the task at the given placement time,
                            # and place it on the WorkerPool.
                            worker_pool = self._worker_pools.get_worker_pool(
                                task_placement.worker_pool_id
                                )
                            assert (
                                worker_pool is not None
                            ), f"No WorkerPool found with ID: {task_placement.worker_pool_id}."
                            
                            # Display worker pool utilization before placing task
                            # Cumulate the resources from all the WorkerPools
                            for worker_pool in self._worker_pools.worker_pools:
                                worker_pool_resources = worker_pool.resources
                                for resource_name in set(
                                    map(lambda value: value[0].name, worker_pool_resources.resources)
                                ):
                                    resource = Resource(name=resource_name, _id="any")
                                    self._logger.info(
                                        f"{sim_time},WORKER_POOL_UTILIZATION,{worker_pool.id},"
                                        f"{resource_name},"
                                        f"{worker_pool_resources.get_allocated_quantity(resource)},"
                                        f"{worker_pool_resources.get_available_quantity(resource)}"
                                    )
                                    
                            # Perform worker pool step
                            self._logger.info(
                                "[%s] Need to perform a step before place_task() for %s.",
                                sim_time,
                                task,
                            )
                            completed_tasks = self.PerformWorkerPoolStep(sim_time=sim_time)

                            # Finish all tasks that have now completed
                            for completed_task in completed_tasks:
                                self.CleanupTaskExecution(
                                    task=completed_task,
                                    sim_time=sim_time
                                    )
                            
                            # Place the task on the worker pool
                            if self._scheduler_is_task_type:
                                success = True
                            else:
                                success = worker_pool.place_task(
                                    task,
                                    execution_strategy=task_placement.execution_strategy,
                                    worker_id=task_placement.worker_id,
                                )
                            if success:
                                task.start(sim_time)
                                self._logger.info(
                                    "[%s] Successfully started task: %s on worker_pool: %s",
                                    sim_time,
                                    task,
                                    worker_pool,
                                )
                                # resources = placement.execution_strategy.resources
                                placements.append(
                                    erdos_scheduler_pb2.Placement(
                                        worker_id=task_placement.worker_id,
                                        application_id=request.id,
                                        task_id=task_placement.task.stage_id,
                                        cores=1,
                                    )
                                )

                                # Add to delete list for clearing placement after it has been released
                                to_delete.append((request.id, task))
                                self._logger.debug(
                                    "[%s] Added tuple (%s, %s) to to_delete list.",
                                    sim_time,
                                    request.id,
                                    task,
                                )
                                
                                # Add task_placement to executed_placements since it is now complete
                                self._executed_placements[task] = task_placement
                            else:
                                self._logger.warning(
                                    "[%s] Could not start task: %s on worker_id: %s and execution strategy: %s",
                                    sim_time,
                                    task,
                                    task_placement.worker_id,
                                    task_placement.execution_strategy,
                                )
                        except ValueError as e:
                            self._logger.error(f"[{sim_time}] start() errored for task: {task}")
                            self._logger.error(f"[{sim_time}] Error: {e}")

        # Remove issued placements from self._placements
        for app_id, task_name in to_delete:
            del self._placements[app_id][task_name]
            self._logger.info(
                "[%s] Removed placement (app_id=%s, task_name=%s) from self._placements",
                sim_time,
                app_id,
                task_name,
                )
        
        self._logger.info(
            "[%s] Constructed %s placements for application with ID %s.",
            sim_time,
            len(placements),
            request.id,
        )

        # Run the scheduler since the Workload has changed.
        await self.run_scheduler()

        return erdos_scheduler_pb2.GetPlacementsResponse(
            success=True,
            placements=placements,
            message=f"[{sim_time}] Constructed {len(placements)} "
            f"placements.",
        )

    # Function to pop tasks from queue based on actual completion time
    async def PopTasksBasedOnTime(self):
        while True:
            if not self._tasks_marked_for_completion.empty():
                # Get the top item from the priority queue
                top_item = self._tasks_marked_for_completion._queue[0][1]

                # Check if top item's timestamp is reached or passed by current time
                current_time = EventTime(int(time.time()), EventTime.Unit.US) - self._initialization_time
                if top_item.timestamp <= current_time.time:
                    # Pop the top item
                    popped_item = self._tasks_marked_for_completion.get()
                    self._logger.info(
                        "[%s] Removing task from pending completion queue. "
                        "Task details: %s",
                        current_time,
                        popped_item.task,
                    )

                    # Display worker pool utilization before removing task
                    # Cumulate the resources from all the WorkerPools
                    for worker_pool in self._worker_pools.worker_pools:
                        worker_pool_resources = worker_pool.resources
                        for resource_name in set(
                            map(lambda value: value[0].name, worker_pool_resources.resources)
                        ):
                            resource = Resource(name=resource_name, _id="any")
                            self._logger.info(
                                f"{current_time},WORKER_POOL_UTILIZATION,{worker_pool.id},"
                                f"{resource_name},"
                                f"{worker_pool_resources.get_allocated_quantity(resource)},"
                                f"{worker_pool_resources.get_available_quantity(resource)}"
                            )

                    # Perform worker pool step
                    self._logger.info(
                        "[%s] Need to perform a step before remove_task() for %s.",
                        current_time,
                        popped_item.task,
                        )
                    completed_tasks = self.PerformWorkerPoolStep(sim_time=current_time)
                    # TODO: (DG) For simplicity, we only pop cleanup task state for a single 
                    # popped-item in the loop at once. Later, we could cleanup all identified
                    # completed tasks here.

                    if popped_item.task.state == TaskState.COMPLETED:
                        # It means that the task state was already cleaned up after another
                        # invocation of PerformWorkerPoolStep. Can skip here then.
                        self._logger.info(
                            "[%s] Task %s already in COMPLETED state while processing "
                            "in PopTasksBasedOnTime.",
                            current_time,
                            popped_item.task,
                        )
                    else:
                        self._logger.info(
                            "[%s] PopTasksBasedOnTime invoking CleanupTaskExecution "
                            "for task %s",
                            current_time,
                            popped_item.task,
                        )
                        self.CleanupTaskExecution(task=popped_item.task,
                                                  sim_time=current_time)

                    # # Free the resources on the worker pool for the completed task
                    # task_placed_at_worker_pool = self._worker_pools.get_worker_pool(
                    #     popped_item.task.worker_pool_id
                    # )
                    # task_placed_at_worker_pool.remove_task(
                    #     current_time=current_time, task=popped_item.task
                    #     )

                    # # Mark the Task as completed.
                    # # Also release the task from the scheduler service
                    # popped_item.task.update_remaining_time(EventTime.zero())
                    # popped_item.task.finish(current_time)

                    # # TODO: (DG) Check change here
                    # released_tasks, cancelled_tasks = self._workload.notify_task_completion(
                    #     task=popped_item.task,
                    #     finish_time=current_time)
                    
                    # # TODO: (DG) Check change here
                    # for new_released_task in released_tasks:
                    #     new_released_task.release(current_time)
                    
                    # # TODO: Might do for cancelled too

                    # # Mark task graph completed
                    # task_graph = self._workload.get_task_graph(popped_item.task.task_graph)
                    # if task_graph is None:
                    #     self._logger.error(f"[{current_time}] Taskgraph for task {popped_item.task} is None")
                    #     raise RuntimeError(f"[{current_time}] Taskgraph for task {popped_item.task} is None")
                    # if task_graph.is_complete():
                    #     self._logger.info(f"[{current_time}] Finished task_graph {task_graph.name}")
                    #     if task_graph.deadline < current_time:
                    #         self._logger.info(f"[{current_time}] Missed deadline for task_graph {task_graph.name}")
                    #         self._total_taskgraphs_missed += 1
                    #     else:
                    #         self._logger.info(f"[{current_time}] Met deadline for task_graph {task_graph.name}")
                    #         self._total_taskgraphs_met += 1
                    #     self._logger.info(
                    #         "[%s] RUN_STATS (registered, met, missed, cancelled): %s, %s, %s, %s",
                    #         current_time,
                    #         self._total_taskgraphs_registered,
                    #         self._total_taskgraphs_met,
                    #         self._total_taskgraphs_missed,
                    #         self._total_taskgraphs_cancelled,
                    #         )

                    # Run the scheduler since the Workload has changed.
                    await self.run_scheduler()
                
                else:
                    # If the top item's timestamp hasn't been reached yet,
                    # sleep for a short duration
                    await asyncio.sleep(0.1)  # TODO: Can adjust value, curr=0.1s
            else:
                # If the queue is empty, sleep for a short duration
                await asyncio.sleep(0.1)  # TODO: Can adjust value, curr=0.1s

    def PerformWorkerPoolStep(self, sim_time):
        # Get time elapsed since last step up time
        time_elapsed_since_last_step = (
            sim_time - self._last_step_up_time
        )
        
        # step up all tasks on the worker-pool to reflect correct remaining time
        self._logger.info(
            "[%s] Stepping for %s timesteps.",
            sim_time,
            time_elapsed_since_last_step,
        )
        for worker_pool in self._worker_pools.worker_pools:
            completed_tasks = worker_pool.step(
                self._last_step_up_time, time_elapsed_since_last_step)
            for task in completed_tasks:
                self._logger.info(
                    "[%s] Task %s was now found complete.",
                    sim_time,
                    task,
                )

        # Update _last_step_up_time
        self._last_step_up_time = sim_time

        return completed_tasks
    
    def CleanupTaskExecution(self, task, sim_time):
        self._logger.info(
            "[%s] Cleaning up task execution for task %s.",
            sim_time,
            task,
            )
        
        # Free the resources on the worker pool for the completed task
        task_placed_at_worker_pool = self._worker_pools.get_worker_pool(
            task.worker_pool_id
        )
        task_placed_at_worker_pool.remove_task(
            current_time=sim_time, task=task
            )

        # Mark the Task as completed.
        # Also release the task from the scheduler service
        task.update_remaining_time(EventTime.zero())
        task.finish(sim_time)

        released_tasks, cancelled_tasks = self._workload.notify_task_completion(
            task=task,
            finish_time=sim_time)
        
        for new_released_task in released_tasks:
            new_released_task.release(sim_time)
        
        # TODO: Might do for cancelled too

        # Mark task graph completed
        task_graph = self._workload.get_task_graph(task.task_graph)
        if task_graph is None:
            self._logger.error(f"[{sim_time}] Taskgraph for task {task} is None")
            raise RuntimeError(f"[{sim_time}] Taskgraph for task {task} is None")
        if task_graph.is_complete():
            self._logger.info(f"[{sim_time}] Finished task_graph {task_graph.name}")
            if task_graph.deadline < sim_time:
                self._logger.info(f"[{sim_time}] Missed deadline for task_graph {task_graph.name}")
                self._total_taskgraphs_missed += 1
            else:
                self._logger.info(f"[{sim_time}] Met deadline for task_graph {task_graph.name}")
                self._total_taskgraphs_met += 1
            self._logger.info(
                "[%s] RUN_STATS (registered, met, missed, cancelled): %s, %s, %s, %s",
                sim_time,
                self._total_taskgraphs_registered,
                self._total_taskgraphs_met,
                self._total_taskgraphs_missed,
                self._total_taskgraphs_cancelled,
                )


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
