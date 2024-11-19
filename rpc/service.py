import time
import asyncio
from concurrent import futures
from collections import namedtuple
from urllib.parse import urlparse
from typing import Optional
from enum import Enum

# TODO: refactor out the need to import main to get common flags
import main
from schedulers import EDFScheduler
from simulator import Simulator, Event, EventTime, EventType
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources, Workload, TaskGraph
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


class DataLoader(Enum):
    TPCH = "tpch"


class WorkloadLoader(BaseWorkloadLoader):
    def __init__(self) -> None:
        self._workload = Workload.empty()

    def add_task_graph(self, task_graph: TaskGraph):
        self._workload.add_task_graph(task_graph)

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        return self._workload


RegisteredTaskGraph = namedtuple("RegisteredTaskGraph", ["graph", "stage_id_mapping"])


class Servicer(erdos_scheduler_pb2_grpc.SchedulerServiceServicer):
    def __init__(self) -> None:
        self._logger = setup_logging(
            name=__name__,
            log_dir=FLAGS.log_dir,
            log_file=FLAGS.log_file_name,
            log_level=FLAGS.log_level,
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

        # There is a bug in the simulator that causes tasks to be placed
        # in the past if the scheduler runtime is not zero
        self._scheduler = EDFScheduler(runtime=EventTime(0, EventTime.Unit.US))

        self._registered_task_graphs = {}
        # TODO: (Dhruv) Can we get the currently active task graphs directly from the workload object?
        self._active_task_graphs = set()

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
        self._initialization_time = EventTime(t, EventTime.Unit.S)
        stime = self.__stime()

        parsed_uri = urlparse(self._master_uri)
        worker_pool = WorkerPool(
            name=f"WorkerPool_{parsed_uri.netloc}",
            _logger=self._logger,
        )
        self._workload_loader = WorkloadLoader()
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
        pass

    async def DeregisterDriver(self, request, context):
        pass

    async def RegisterTaskGraph(self, request, context):
        stime = self.__stime()

        if not self.__framework_registered():
            msg = f"[{stime}] Trying to register a task graph (id={request.id}, name={request.name}) but no framework has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        if request.id in self._registered_task_graphs:
            msg = f"[{stime}] The task graph (id={request.id}, name={request.name}) is already registered"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        # We only support TPCH queries for now
        if request.name.startswith("TPCH Query"):
            # Parse request name
            query_parts = request.name.split()
            if len(query_parts) != 3 and len(query_parts) != 5:
                msg = f"[{stime}] Invalid TPCH query request"
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False, message=msg, num_executors=0
                )
            query_num = int(query_parts[2])
            if len(query_parts) == 5:
                dataset_size = int(query_parts[3])
                max_executors_per_job = int(query_parts[4])
            else:
                dataset_size = FLAGS.tpch_dataset_size
                max_executors_per_job = FLAGS.tpch_max_executors_per_job

            # Convert request.dependencies to [{name: int, children: [int]}]
            dependencies = []
            for dep in request.dependencies:
                dependencies.append(
                    {
                        "name": int(dep.key.id),
                        "children": [int(c) for c in dep.children_ids],
                    }
                )

            # Construct the task graph
            try:
                task_graph, stage_id_mapping = self._data_loaders[
                    DataLoader.TPCH
                ].make_task_graph(
                    id=id,
                    query_num=query_num,
                    release_time=stime,
                    dependencies=dependencies,
                    dataset_size=dataset_size,
                    max_executors_per_job=max_executors_per_job,
                    runtime_unit=EventTime.Unit.S,
                )
            except Exception as e:
                msg = f"[{stime}] Failed to load TPCH query {query_num}. Exception: {e}"
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False, message=msg, num_executors=0
                )
        else:
            msg = f"[{stime}] The service only supports TPCH queries"
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        self._registered_task_graphs[request.id] = RegisteredTaskGraph(
            task_graph, stage_id_mapping
        )
        msg = f"[{stime}] Registered task graph '{task_graph.name}' successfully"

        # Add the task graph to the active task graphs if registration is successful
        self._active_task_graphs.add(request.id)

        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=msg,
            num_executors=FLAGS.spark_app_num_initial_executors,
        )

    async def RegisterEnvironmentReady(self, request, context):
        stime = self.__stime()

        if not self.__framework_registered():
            msg = f"[{stime}] Trying to notify that the environment is ready for task graph (id={request.id}) but no framework is registered yet"
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=msg,
            )

        # TODO: check if the task graph exists

        task_graph = self._registered_task_graphs[request.id].graph

        self._workload_loader.add_task_graph(task_graph)
        self._simulator._event_queue.add_event(
            Event(
                event_type=EventType.UPDATE_WORKLOAD,
                time=stime,
            )
        )

        msg = f"[{stime}] Successfully marked environment as ready for task graph '{task_graph.name}'"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
            success=True,
            message=msg,
        )

    async def RegisterWorker(self, request, context):
        stime = self.__stime()

        if not self.__framework_registered():
            msg = f"[{stime}] Trying to register a worker (id={request.id}, name={request.name}) but no framework is registered yet"
            return erdos_scheduler_pb2.RegisterWorkerResponse(
                success=False, message=msg
            )

        # TODO(Sukrit): Right now, we drop the memory requirements, we should use
        # them to do multi-dimensional packing using STRL.

        cpu_resource = Resource(name="Slot")
        worker_resources = Resources(
            resource_vector={
                # TODO(elton): handle override worker cpu count?
                cpu_resource: request.cores,
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

        # TODO (Dhruv): Can add check to verify that framework and worker are registered

        # Check if the task graph is registered
        if request.id not in self._registered_task_graphs:
            msg = (
                f"[{stime}] Task graph with id {request.task_graph_id} not registered."
            )
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False,
                message=msg,
            )

        # Check if the task graph is active
        if request.id not in self._active_task_graphs:
            msg = f"[{stime}] Task graph with id {request.task_graph_id} not active."
            return erdos_scheduler_pb2.GetPlacementsResponse(
                success=False,
                message=msg,
            )

        task_graph = self._registered_task_graphs[request.id].graph
        placements = self._simulator.get_current_placements_for_task_graph(
            task_graph.name
        )
        self._logger.info(
            f"Received the following placements for '{task_graph.name}': {placements}"
        )

        return erdos_scheduler_pb2.GetPlacementsResponse(
            success=True,
            message=f"Placements for taskgraph {request.id} returned successfully.",
        )

    async def NotifyTaskCompletion(self, request, context):
        pass

    async def _tick_simulator(self):
        while True:
            if self._simulator is not None:
                stime = self.__stime()
                self._logger.debug(f"[{stime}] Simulator tick")
                self._simulator.tick(until=stime)
            else:
                print("Simulator instance is None")
            await asyncio.sleep(1)

    def __stime(self) -> EventTime:
        """
        Time as viewed by the service. Starts when a framework is registered
        and ends when it is deregistered.
        """
        if self._initialization_time is None:
            return EventTime.invalid()
        ts = int(time.time())
        ts = EventTime(ts, EventTime.Unit.S)
        return ts - self._initialization_time

    def __framework_registered(self):
        return self._simulator is not None

    def __get_worker_pool(self):
        # Simulator maintains only one worker pool, so this should be fine
        return next(iter(self._simulator._worker_pools.worker_pools))


async def serve(server):
    await server.start()
    print("Initialized ERDOS RPC Service on port", FLAGS.port)
    await server.wait_for_termination()


def main(_argv):
    loop = asyncio.get_event_loop()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=FLAGS.max_workers))
    servicer = Servicer()
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
