import threading
import sys
import time
import asyncio
from concurrent import futures
from urllib.parse import urlparse
from typing import Optional, Dict, Callable, Tuple
from enum import Enum
from dataclasses import dataclass

# TODO: refactor out the need to import main to get common flags
import main
from simulator import Simulator, Event, EventTime, EventType
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    Resource,
    Resources,
    Workload,
    TaskGraph,
    TaskState,
    Placement,
    JobGraph,
)
from data import BaseWorkloadLoader
from data.tpch_loader import TpchLoader
from utils import setup_logging, setup_csv_logging
from rpc import erdos_scheduler_pb2
from rpc import erdos_scheduler_pb2_grpc

import grpc

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 50051, "Port to serve the ERDOS Scheduling RPC Server on.")
flags.DEFINE_integer(
    "max_workers", 10, "Maximum number of workers to use for the RPC server."
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
flags.DEFINE_integer(
    "spark_app_num_initial_executors",
    10,
    "The initial number of executors that are requested by each Spark application.",
)
flags.DEFINE_bool(
    "override_worker_cpu_count",
    False,
    "If True, worker CPU count will be set to 640 (Cloudlab 20-node cluster CPU count). "
    "This allows us to scale up spark experiments without actually deploying a large "
    "spark cluster.",
)


class DataLoader(Enum):
    TPCH = "tpch"


class WorkloadLoader(BaseWorkloadLoader):
    def __init__(self, _flags) -> None:
        self._workload = Workload.empty(_flags)

    def add_task_graph(self, task_graph: TaskGraph):
        self._workload.add_task_graph(task_graph)

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        return self._workload


@dataclass
class RegisteredApplication:
    """
    Represents a registered application that can be used to generate task
    graphs. It also manages the mapping between Spark stage IDs and canonical
    task IDs.

    A registered application is ready if the `task_graph` attribute is set.

    Attributes:
        gen (Callable[[EventTime], Tuple[TaskGraph, Dict[int,int]]]):
            A function that takes a release time and outputs:
            - A task graph
            - A mapping from Spark stage IDs to canonical task IDs

        task_graph (TaskGraph, optional):
            The generated task graph for the application. Defaults to None.

    Methods:
        generate_task_graph(release_time: EventTime):
            Sets the `task_graph` attribute by generating a task graph for a
            given `release_time`.

        spark_task_id(task_id: int):
            Returns the canonical task ID corresponding to a Spark stage ID.

        canonical_task_id(stage_id: int):
            Returns the Spark stage ID corresponding to a canonical task ID.
    """

    gen: Callable[[EventTime], Tuple[TaskGraph, Dict[int, int]]]
    task_graph: TaskGraph = None

    _forward: Dict[int, int] = None  # spark stage id => canonical task id
    _backward: Dict[int, int] = None  # canonical task id => spark stage id
    _last_gen: EventTime = None

    def __init__(self, gen):
        self.gen = gen

    def generate_task_graph(self, release_time: EventTime):
        task_graph, stage_id_mapping = self.gen(release_time)
        self.task_graph = task_graph
        self._forward = stage_id_mapping
        self._backward = {v: k for k, v in self._forward.items()}
        self._last_gen = release_time

    def spark_task_id(self, task_id: int):
        return self._backward[task_id]

    def canonical_task_id(self, stage_id: int):
        return self._forward[stage_id]


class Servicer(erdos_scheduler_pb2_grpc.SchedulerServiceServicer):
    def __init__(self, server) -> None:
        self._server = server

        # Override some flags

        # Enable orchestrated mode
        FLAGS.orchestrated = True
        # Set scheduler runtime to zero
        FLAGS.scheduler_runtime = 0

        self._logger = setup_logging(
            name=__name__,
            log_dir=FLAGS.log_dir,
            log_file=FLAGS.log_file_name,
            log_level=FLAGS.log_level,
            fmt="[%(asctime)s] {%(funcName)s:%(lineno)d} - %(message)s",
        )
        self._csv_logger = setup_csv_logging(
            name=__name__,
            log_dir=FLAGS.log_dir,
            log_file=FLAGS.csv_file_name,
        )
        for flag_name in FLAGS:
            self._csv_logger.debug(
                f"input_flag,{flag_name},{getattr(FLAGS, flag_name)}"
            )

        self._master_uri = None
        self._initialization_time = None
        self._data_loaders = {}
        self._data_loaders[DataLoader.TPCH] = TpchLoader(
            path=FLAGS.tpch_query_dag_spec,
            flags=FLAGS,
        )
        self._simulator = None
        self._workload_loader = None

        # Instantiate the scheduler based on the given flag.
        self._scheduler = None
        if FLAGS.scheduler == "FIFO":
            from schedulers import FIFOScheduler

            self._scheduler = FIFOScheduler(
                preemptive=FLAGS.preemption,
                runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
                enforce_deadlines=FLAGS.enforce_deadlines,  # TODO: (DG) Check why this isnt passed in the simulator
                _flags=FLAGS,
            )
        elif FLAGS.scheduler == "EDF":
            from schedulers import EDFScheduler

            self._scheduler = EDFScheduler(
                preemptive=FLAGS.preemption,
                runtime=EventTime(FLAGS.scheduler_runtime, EventTime.Unit.US),
                enforce_deadlines=FLAGS.enforce_deadlines,
                _flags=FLAGS,
            )
        elif FLAGS.scheduler == "TetriSched":
            from schedulers import TetriSchedScheduler

            finer_discretization = FLAGS.finer_discretization_at_prev_solution
            self._scheduler = TetriSchedScheduler(
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
        else:
            raise ValueError(f"Unknown scheduler {FLAGS.scheduler}.")

        # TODO: Items in _registered_applications are never deleted right now, needs to be handled.
        self._registered_applications = {}
        self._registered_app_drivers = (
            {}
        )  # Spark driver id differs from taskgraph name (application id)
        self._received_shutdown = False
        self._shutting_down = False
        self._lock = threading.Lock()

        super().__init__()

    async def RegisterFramework(self, request, context):
        stime = self.__stime()

        if self.__framework_registered():
            msg = f"[{stime}] Framework already registered at the address {self._master_uri} at timestamp {self._initialization_time}"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=msg,
            )

        t = int(time.time())
        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = EventTime(t, EventTime.Unit.US)
        stime = self.__stime()

        parsed_uri = urlparse(self._master_uri)
        worker_pool = WorkerPool(
            name=f"WorkerPool_{parsed_uri.netloc}",
            _logger=self._logger,
        )
        self._workload_loader = WorkloadLoader(FLAGS)

        self._simulator = Simulator(
            scheduler=self._scheduler,
            worker_pools=WorkerPools(
                [worker_pool]
            ),  # Maintain only one worker pool in the simulator
            workload_loader=self._workload_loader,
            _flags=FLAGS,
        )

        msg = f"[{stime}] Registered the framework '{framework_name}' with URI {self._master_uri} at UNIX time {self._initialization_time.time}"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterFrameworkResponse(success=True, message=msg)

    async def DeregisterFramework(self, request, context):
        stime = self.__stime()

        if not self.__framework_registered():
            msg = f"[{stime}] Trying to deregister a framework at {request.uri} but no framework has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message=msg
            )

        if self._master_uri != request.uri:
            msg = f"[{stime}] Trying to deregister the framework at {request.uri} but the registered framework is at {self._master_uri}"
            self._logger.error(msg)
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message=msg
            )

        self._initialization_time = None
        self._master_uri = None
        self._workload_loader = None
        self._simulator = None
        msg = f"[{stime}] Successfully deregistered the framework at {request.uri}"
        self._logger.info(msg)
        return erdos_scheduler_pb2.DeregisterFrameworkResponse(
            success=True, message=msg
        )

    async def RegisterDriver(self, request, context):
        stime = self.__stime()

        if not self.__worker_registered():
            msg = f"[{stime}] Failed to register driver (id={request.id}) because no worker has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message=msg,
            )

        if request.id in self._registered_app_drivers:
            msg = f"[{stime}] Driver with id '{request.id}' is already registered"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message=msg,
                worker_id=self.__get_worker_id(),
            )

        # TODO: Update the registered_app_drivers to map the driver id to
        # application id once the taskgraph is registered.
        self._registered_app_drivers[request.id] = None

        msg = (
            f"[{stime}] Successfully registered driver {request.id} for an application."
        )
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterDriverResponse(
            success=True,
            message=msg,
            worker_id=self.__get_worker_id(),
        )

    async def DeregisterDriver(self, request, context):
        stime = self.__stime()

        if request.id not in self._registered_app_drivers:
            msg = f"[{stime}] Driver id '{request.id}' is not registered or does not exist"
            self._logger.error(msg)
            return erdos_scheduler_pb2.DeregisterDriverResponse(
                success=False,
                message=msg,
            )

        # TODO: Dummy mapping from driver to task graph (application), so task_graph_name is None.
        # Deletion of taskgraph from registered_applications and driver from registered_app_drivers should be done carefully.
        task_graph_name = self._registered_app_drivers[request.id]
        del self._registered_app_drivers[request.id]

        with self._lock:
            log_stats_event = Event(
                event_type=EventType.LOG_STATS,
                time=self.__stime(),
            )
            self._simulator._event_queue.add_event(log_stats_event)

        msg = f"[{stime}] Successfully de-registered driver with id {request.id} for task graph {task_graph_name}"
        self._logger.info(msg)

        if len(self._registered_app_drivers) == 0 and self._received_shutdown:
            self._logger.info(f"[{stime}] The last driver has been deregistered; finishing simulation")
            # Signals _tick_simulator() to stop.  Shouldn't be
            # necessary in principle because after the with block
            # ends, there shouldn't be any more events left to run,
            # but doesn't hurt.
            self._shutting_down = True
            with self._lock:
                self._simulator.simulate()
            await self._server.stop(0)

        return erdos_scheduler_pb2.DeregisterDriverResponse(
            success=True,
            message=msg,
        )

    async def RegisterTaskGraph(self, request, context):
        stime = self.__stime()

        if not self.__worker_registered():
            msg = f"[{stime}] Failed to register task graph (id={request.id}, name={request.name}) because no worker has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        if request.id in self._registered_applications:
            msg = f"[{stime}] The task graph (id={request.id}, name={request.name}) is already registered"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        # We only support TPCH queries for now
        if request.name.startswith("TPCH Query"):
            # Parse request name
            query_parts = request.name.split()
            match query_parts:
                case _, _, query_num, index, dataset_size, max_executors_per_job:
                    query_num = int(query_num)
                    dataset_size = int(dataset_size)
                    max_executors_per_job = int(max_executors_per_job)
                case _, _, query_num, dataset_size, max_executors_per_job:
                    query_num = int(query_num)
                    # default index counts up from 0; incorrect if
                    # Spark receives jobs out of order
                    index = str(len(self._registered_applications))
                    dataset_size = int(dataset_size)
                    max_executors_per_job = int(max_executors_per_job)
                case _, _, query_num:
                    query_num = int(query_num)
                    index = str(len(self._registered_applications))
                    dataset_size = FLAGS.tpch_dataset_size
                    max_executors_per_job = FLAGS.tpch_max_executors_per_job
                case _:
                    msg = f"[{stime}] Invalid TPCH query request"
                    return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                        success=False, message=msg, num_executors=0
                    )

            # Convert request.dependencies to [{name: int, children: [int]}]
            dependencies = []
            for dep in request.dependencies:
                dependencies.append(
                    {
                        "name": int(dep.key.id),
                        "children": [int(c) for c in dep.children_ids],
                    }
                )

            # Create a job graph
            self._logger.debug(str((query_num, index, dataset_size, max_executors_per_job)))
            try:
                job_graph, stage_id_mapping = self._data_loaders[
                    DataLoader.TPCH
                ].make_job_graph(
                    id=index,
                    query_num=query_num,
                    dependencies=dependencies,
                    dataset_size=dataset_size,
                    max_executors_per_job=max_executors_per_job,
                    runtime_unit=EventTime.Unit.US,
                )
            except Exception as e:
                msg = f"[{stime}] Failed to load TPCH query {query_num}. Exception: {e}"
                self._logger.error(msg)
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False, message=msg, num_executors=0
                )

            if not self.__can_accomodate_task_graph(job_graph):
                msg = f"[{stime}] The worker Pool cannot accomodate the task graph '{request.id}'"
                self._logger.error(msg)
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False, message=msg, num_executors=0
                )

            def gen(release_time):
                task_graph = job_graph.get_next_task_graph(
                    start_time=release_time,
                    _flags=FLAGS,
                )
                return task_graph, stage_id_mapping

        else:
            msg = f"[{stime}] The service only supports TPCH queries"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        self._registered_applications[request.id] = RegisteredApplication(gen=gen)

        msg = f"[{stime}] Registered task graph '{request.id}' successfully"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=msg,
            num_executors=FLAGS.spark_app_num_initial_executors,
        )

    async def RegisterEnvironmentReady(self, request, context):
        stime = self.__stime()

        if request.id not in self._registered_applications:
            msg = f"[{stime}] Task graph of id '{request.id}' is not registered or does not exist"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=msg,
            )

        r = self._registered_applications[request.id]
        r.generate_task_graph(stime)

        with self._lock:
            self._simulator._workload.add_task_graph(r.task_graph)
            self._simulator._current_task_graph_placements[r.task_graph.name] = {}

            for task in r.task_graph.get_releasable_tasks():
                task_release_event = Event(
                    event_type=EventType.TASK_RELEASE,
                    time=self.__stime(),
                    task=task,
                )
                self._logger.info(
                    f"[{stime}] Added event {task_release_event} to the simulator's event queue",
                )
                self._simulator._event_queue.add_event(task_release_event)

            scheduler_start_event = Event(
                event_type=EventType.SCHEDULER_START,
                time=self.__stime(),
            )
            self._simulator._event_queue.add_event(scheduler_start_event)
            self._logger.info(
                f"[{stime}] Added event {scheduler_start_event} to the simulator's event queue"
            )

        msg = f"[{stime}] Successfully marked environment as ready for task graph '{r.task_graph.name}'"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
            success=True,
            message=msg,
        )

    async def RegisterWorker(self, request, context):
        stime = self.__stime()

        if not self.__framework_registered():
            msg = f"[{stime}] Trying to register a worker (id={request.id}, name={request.name}) but no framework is registered yet"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterWorkerResponse(
                success=False, message=msg
            )

        # TODO(Sukrit): Right now, we drop the memory requirements, we should use
        # them to do multi-dimensional packing using STRL.

        cpu_resource = Resource(name="Slot")
        worker_resources = Resources(
            resource_vector={
                cpu_resource: (
                    request.cores if not FLAGS.override_worker_cpu_count else 640
                )
            },
            _logger=self._logger,
        )
        worker = Worker(
            name=request.id,
            resources=worker_resources,
            _logger=self._logger,
        )

        self.__get_worker_pool().add_workers([worker])

        msg = f"[{stime}] Registered worker (id={request.id}, name={request.name})"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True,
            message=msg,
            cores=FLAGS.virtualized_cores,
            memory=FLAGS.virtualized_memory * 1024,
        )

    async def GetPlacements(self, request, context):
        stime = self.__stime()

        # Check if the task graph is registered
        if request.id not in self._registered_applications:
            msg = f"[{stime}] Task graph with id '{request.id}' is not registered or does not exist"
            self._logger.error(msg)
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False,
                message=msg,
            )

        r = self._registered_applications[request.id]

        if r.task_graph is None:
            msg = f"[{stime}] Task graph '{request.id}' is not ready"
            self._logger.error(msg)
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=True,
                message=msg,
                placements=[],
            )

        if r.task_graph.is_complete():
            msg = f"[{stime}] Task graph '{r.task_graph.name}' is complete. No more placements to provide."
            self._logger.error(msg)
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=True,
                message=msg,
            )

        # A task graph is considered complete if **all** of its **sink** tasks
        # are complete. It is considered cancelled if **any** of its **sink**
        # tasks are cancelled.

        # If the task graph is complete, the Spark application will
        # automatically shut down because it knows that all of its stages have
        # finished executing.

        # Matters get interesting in the presence of task cancellations. The
        # service is aware of which tasks are cancelled.

        # First, even when a task graph is cancelled, the simulator (without
        # orchestration)
        # continues to schedule and execute any tasks that were released into
        # the system. The service, which runs the simulator in orchestrated
        # mode, must emulate this behavior to maintain parity.

        # Second, from Spark's point of view, however, those tasks are still
        # pending placements. So, Spark will continue to periodically invoke
        # `GetPlacements` in the hopes of receiving placements for those
        # cancelled tasks. Left unhandled, the Spark application will loop
        # indefinitely waiting for placements.

        # We _could_ communicate these task cancellations to Spark. Then, we
        # can modify the DAGScheduler to invoke GetPlacements until all of its
        # stages have either finished executing or have been cancelled, after
        # which it can safely terminate the application.

        # However, we run into an issue due to VIRTUAL tasks. When a task is
        # cancelled, the simulator invokes `TaskGraph.cancel(task)`.
        # `TaskGraph.cancel(task)` traverses the tree rooted at `task`
        # depth-first, cancelling tasks along the way until it finds the first
        # terminal task. As a consequence, it is possible for the tree rooted
        # at a cancelled task to have VIRTUAL tasks inside of it. These
        # virtual tasks will never receive placements because they are not
        # releasable. So, it is possible for the Spark application to stall on
        # `GetPlacements` waiting on placements for these virtual tasks.

        # Since the service knows the state of each task, it is easy then for
        # the service to determine when the Spark application should terminate
        # in the presence of task cancellations.

        # So, instead of communicating task cancellations, we communicate when
        # the Spark application should terminate.
        #
        # The first check makes sure all tasks are either CANCELLED,
        # COMPLETED, or VIRTUAL. We check for all tasks because it is possible
        # that the simulator is processing released and scheduled tasks. If we
        # terminate early, then we will never receive `NotifyTaskCompletion`s
        # for those tasks (because the Spark application was terminated),
        # which then results in those tasks never getting removed from the
        # worker pool.
        #
        # The second check makes sure that the task graph is indeed cancelled.
        # We have this additional guard because at the start all tasks are
        # VIRTUAL and we don't want to terminate the application then.
        
        if r.task_graph.is_cancelled():
            self._logger.error(f"[{stime}] Task graph '{r.task_graph.name}' is in state cancelled.")

        should_terminate = all(
            task.state
            in (
                TaskState.CANCELLED,
                TaskState.COMPLETED,
                TaskState.VIRTUAL,
            )
            for task in r.task_graph
        ) and (r.task_graph.is_cancelled())
        if should_terminate:
            msg = f"[{stime}] Task graph '{r.task_graph.name}' was cancelled and simulator has processed all released/ scheduled tasks. Terminating it since it has no more placements to provide."
            self._logger.error(msg)
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=True,
                message=msg,
                terminate=True,
            )
        elif r.task_graph.is_cancelled() and not should_terminate:
            msg = f"[{stime}] Task graph '{r.task_graph.name}' was cancelled but simulator is still processing some released/ scheduled tasks. Will provide placements."
            self._logger.error(msg)
        else:
            msg = f"[{stime}] Task graph '{r.task_graph.name}' is actively running. Will provide placements."
            self._logger.info(msg)

        with self._lock:
            sim_placements = self._simulator.get_current_placements_for_task_graph(
                r.task_graph.name
            )

        placements = []
        for placement in sim_placements:
            if placement.task.state != TaskState.RUNNING:
                self._logger.debug(f"[{stime}] Skipping placement: {placement}")
                continue

            worker_id = (
                self.__get_worker_id()
                if placement.placement_type == Placement.PlacementType.PLACE_TASK
                else "None"
            )
            task_id = r.spark_task_id(placement.task.name)
            cores = (
                sum(x for _, x in placement.execution_strategy.resources.resources)
                if placement.placement_type == Placement.PlacementType.PLACE_TASK
                else 0
            )

            if placement.placement_type not in (Placement.PlacementType.PLACE_TASK,):
                raise NotImplementedError

            placements.append(
                {
                    "worker_id": worker_id,
                    "application_id": request.id,
                    "task_id": task_id,
                    "cores": cores,
                },
            )
            
        msg = f"[{stime}] Returning the following placements {placements} for task graph '{request.id}'."
        self._logger.info(msg)
        return erdos_scheduler_pb2.GetPlacementsResponse(
            success=True,
            message=msg,
            placements=placements,
        )

    async def NotifyTaskCompletion(self, request, context):
        stime = self.__stime()

        # Check if the task graph is registered
        if request.application_id not in self._registered_applications:
            msg = f"[{stime}] Task graph with id '{request.id}' is not registered or does not exist"
            self._logger.error(msg)
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=msg,
            )

        r = self._registered_applications[request.application_id]
        task = r.task_graph.get_task(r.canonical_task_id(request.task_id))
        if task is None:
            msg = f"[{stime}] Task '{request.task_id}' does not exist in the task graph '{r.task_graph.name}'"
            self._logger.error(msg)
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=msg,
            )

        if task.state != TaskState.RUNNING:
            msg = f"[{stime}] Received task completion notification for task '{request.task_id}' (mapped to '{r.canonical_task_id(request.task_id)}') of '{r.task_graph.name}' but it is not running"
            self._logger.error(msg)
            return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
                success=False,
                message=msg,
            )

        # HACK: The worker pool doesn't step every tick (probably should). So, the task.remaining_time is not accurate. We compute actual_task_completion then by getting the runtime from the profile,
        actual_task_completion_time = (
            task.start_time + task.slowest_execution_strategy.runtime
        )

        with self._lock:
            # NOTE: Although the actual_task_completion_time works for task completion notifications that arrive early, it is
            # inaccurate for task completion notifications that occur past that time. Thus, a max of the current and actual completion time
            # is taken to ensure that the task is marked completed at the correct time.
            task_finished_event = Event(
                event_type=EventType.TASK_FINISHED,
                time=max(actual_task_completion_time, self.__stime()),
                task=task,
            )
            self._simulator._event_queue.add_event(task_finished_event)
            self._logger.info(
                f"[{stime}] Adding event {task_finished_event} to the simulator's event queue"
            )
            if actual_task_completion_time < self.__stime():
                self._logger.error(
                    f"[{stime}] Task '{request.task_id}' of task graph '{r.task_graph.name}' had exceeded its runtime by {self.__stime() - actual_task_completion_time}")

            scheduler_start_event = Event(
                event_type=EventType.SCHEDULER_START,
                time=max(
                    actual_task_completion_time.to(EventTime.Unit.US),
                    self.__stime(),
                ),
            )
            self._simulator._event_queue.add_event(scheduler_start_event)
            self._logger.info(
                f"[{stime}] Added event {scheduler_start_event} to the simulator's event queue"
            )

        msg = f"[{stime}] Successfully processed completion of task '{request.task_id}' of task graph '{r.task_graph.name}'"
        self._logger.info(msg)
        return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
            success=True,
            message=msg,
        )

    async def Shutdown(self, request, context):
        self._received_shutdown = True
        return erdos_scheduler_pb2.Empty()

    async def _tick_simulator(self):
        while not self._shutting_down:
            with self._lock:
                if self._simulator is not None:
                    stime = self.__stime()
                    # self._logger.debug(f"[{stime}] Simulator tick")
                    self._simulator.tick(until=stime)
            # else:
            #     print("Simulator instance is None")
            await asyncio.sleep(1)

    def __stime(self) -> EventTime:
        """
        Time as viewed by the service. Starts when a framework is registered
        and ends when it is deregistered.
        """
        if self._initialization_time is None:
            return EventTime.invalid()
        ts = int(time.time())
        # NOTE: The service runs in the US time unit for better compatibility with the simulator.
        # The simulator uses an abstract unit of time, and it is all relative.
        ts = EventTime(ts, EventTime.Unit.US)
        return ts - self._initialization_time

    def __framework_registered(self):
        return self._simulator is not None

    def __worker_registered(self):
        return (
            self.__framework_registered() and len(self.__get_worker_pool().workers) > 0
        )

    def __get_worker_pool(self):
        # Simulator maintains only one worker pool, so this should be fine
        return next(iter(self._simulator._worker_pools.worker_pools))

    def __get_worker(self):
        return self.__get_worker_pool().workers[0]

    def __get_worker_id(self):
        # We return the name here because we register the worker id from
        # Spark as the name of the worker in the worker pool
        return self.__get_worker().name

    def __can_accomodate_task_graph(self, job_graph: JobGraph):
        worker_resources = self.__get_worker().resources
        for job in job_graph:
            for strat in job.execution_strategies:
                for resource, quantity in strat.resources.resources:
                    if worker_resources.get_total_quantity(resource) < quantity:
                        return False
        return True


async def serve(server):
    await server.start()
    print("Initialized ERDOS RPC Service on port", FLAGS.port)
    await server.wait_for_termination()


def main(_argv):
    loop = asyncio.get_event_loop()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=FLAGS.max_workers))
    servicer = Servicer(server)
    erdos_scheduler_pb2_grpc.add_SchedulerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{FLAGS.port}")

    # Schedule the periodic tick_simulator task
    loop.create_task(servicer._tick_simulator())

    try:
        loop.run_until_complete(serve(server))
    except KeyboardInterrupt:
        print("Terminated ERDOS RPC Service")
    finally:
        loop.close()


if __name__ == "__main__":
    app.run(main)
