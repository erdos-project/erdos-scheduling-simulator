import time
import asyncio
from concurrent import futures
from urllib.parse import urlparse
from typing import Optional
from enum import Enum

# TODO: refactor out the need to import main to get common flags
import main
from schedulers import EDFScheduler
from simulator import Simulator, EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources
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
        # TODO: refactor
        self._data_loaders[DataLoader.TPCH] = TpchLoader(
            path=FLAGS.tpch_query_dag_spec, flags=FLAGS
        )
        self._simulator = None
        self._scheduler = EDFScheduler()

        self._registered_task_graphs = {}

    async def RegisterFramework(self, request, context):
        sim_time = self.__sim_time()

        if self.__framework_registered():
            msg = f"[{sim_time}] Framework already registered at the address {self._master_uri} at timestamp {self._initialization_time}"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=msg,
            )

        t = int(time.time())
        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = EventTime(t, EventTime.Unit.US)
        # Update sim_time now that initialization_time is set
        sim_time = self.__sim_time()

        parsed_uri = urlparse(self._master_uri)
        worker_pool = WorkerPool(
            name=f"WorkerPool_{parsed_uri.netloc}",
            _logger=self._logger,
        )
        self._simulator = Simulator(
            scheduler=self._scheduler,
            worker_pools=WorkerPools(
                [worker_pool]
            ),  # Maintain only one worker pool in the simulator
        )

        msg = f"[{sim_time}] Registered the framework '{framework_name}' with URI {self._master_uri} at UNIX time {self._initialization_time.time}"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterFrameworkResponse(success=True, message=msg)

    async def DeregisterFramework(self, request, context):
        sim_time = self.__sim_time()

        if not self.__framework_registered():
            msg = f"Trying to deregister a framework at {request.uri} but no framework has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message=msg
            )

        if self._master_uri != request.uri:
            msg = f"Trying to deregister the framework at {request.uri} but the registered framework is at {self._master_uri}"
            self._logger.error(msg)
            return erdos_scheduler_pb2.DeregisterFrameworkResponse(
                success=False, message=msg
            )

        self._initialization_time = None
        self._master_uri = None
        self._simulator = None
        msg = f"[{sim_time}] Successfully deregistered the framework at {request.uri}"
        self._logger.info(msg)
        return erdos_scheduler_pb2.DeregisterFrameworkResponse(
            success=True, message=msg
        )

    async def RegisterDriver(self, request, context):
        pass

    async def DeregisterDriver(self, request, context):
        pass

    async def RegisterTaskGraph(self, request, context):
        sim_time = self.__sim_time()

        if not self.__framework_registered():
            msg = f"[{sim_time}] Trying to register a task graph (id={request.id}, name={request.name}) but no framework has been registered yet."
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        if request.id in self._registered_task_graphs:
            msg = f"[{sim_time}] The task graph (id={request.id}, name={request.name}) is already registered"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        # We only support TPCH queries for now
        if request.name.startswith("TPCH Query"):
            # Parse request name
            query_parts = request.name.split()
            if len(query_parts) != 3 and len(query_parts) != 5:
                msg = f"[{sim_time}] Invalid TPCH query request"
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
                    release_time=sim_time,
                    dependencies=dependencies,
                    dataset_size=dataset_size,
                    max_executors_per_job=max_executors_per_job,
                )
            except Exception as e:
                msg = f"[{sim_time}] Failed to load TPCH query {query_num}. Exception: {e}"
                return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                    success=False, message=msg, num_executors=0
                )
        else:
            msg = f"[{sim_time}] The service only supports TPCH queries"
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False, message=msg, num_executors=0
            )

        self._registered_task_graphs[request.id] = (task_graph, stage_id_mapping)
        msg = f"[{sim_time}] Registered task graph (id={request.id}, name={request.name}) successfully"

        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=msg,
            num_executors=FLAGS.spark_app_num_initial_executors,
        )

    async def RegisterEnvironmentReady(self, request, context):
        sim_time = self.__sim_time()

        if not self.__framework_registered():
            msg = f"[{sim_time}] Trying to notify that the environment is ready for task graph (id={request.id}, name={request.name}) but no framework is registered yet"
            return erdos_scheduler_pb2.RegisterEnvironmentReadyResponse(
                success=False,
                message=msg,
            )

    async def RegisterWorker(self, request, context):
        sim_time = self.__sim_time()

        if not self.__framework_registered():
            msg = f"[{sim_time}] Trying to register a worker (id={request.id}, name={request.name}) but no framework is registered yet"
            return erdos_scheduler_pb2.RegisterWorkerResponse(
                success=False, message=msg
            )

        # TODO(Sukrit): Right now, we drop the memory requirements, we should use
        # them to do multi-dimensional packing using STRL.

        cpu_resource = Resource(name="Slot_CPU")
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

        msg = f"[{sim_time}] Registered worker (id={request.id}, name={request.name})"

        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True,
            message=msg,
            # TODO(elton): not sure why we need to set this here
            cores=FLAGS.virtualized_cores,
            memory=FLAGS.virtualized_memory * 1024,
        )

    async def GetPlacements(self, request, context):
        pass

    async def NotifyTaskCompletion(self, request, context):
        pass

    def __sim_time(self) -> EventTime:
        if self._initialization_time is None:
            return EventTime.invalid()
        ts = int(time.time())
        ts = EventTime(ts, EventTime.Unit.US)
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
    erdos_scheduler_pb2_grpc.add_SchedulerServiceServicer_to_server(Servicer(), server)
    server.add_insecure_port(f"[::]:{FLAGS.port}")

    try:
        loop.run_until_complete(serve(server))
    except KeyboardInterrupt:
        print("Terminated ERDOS RPC Service")
    finally:
        loop.close()


if __name__ == "__main__":
    app.run(main)
