import asyncio
from concurrent import futures
from urllib.parse import urlparse

# TODO: refactor out the need to import main to get common flags
import main
from schedulers import EDFScheduler
from simulator import Simulator, EventTime
from workers import WorkerPool, WorkerPools
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
        self._simulator = None
        self._scheduler = EDFScheduler()

    async def RegisterFramework(self, request, context):
        if self._simulator:
            msg = f"Framework already registered at the address {self._master_uri} at timestamp {self._initialization_time}"
            self._logger.error(msg)
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=msg,
            )

        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = EventTime(request.timestamp, EventTime.Unit.US)

        parsed_uri = urlparse(self._master_uri)
        worker_pool = WorkerPool(
            name=f"WorkerPool_{parsed_uri.netloc}",
            _logger=self._logger,
        )
        self._simulator = Simulator(
            scheduler=self._scheduler,
            worker_pools=WorkerPools([worker_pool]),
        )

        sim_time = (
            EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
        )
        msg = f"[{sim_time}] Registered the framework '{framework_name}' with URI {self._master_uri} at {self._initialization_time.time}"
        self._logger.info(msg)
        return erdos_scheduler_pb2.RegisterFrameworkResponse(success=True, message=msg)

    async def DeregisterFramework(self, request, context):
        if not self._simulator:
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

        sim_time = (
            EventTime(request.timestamp, EventTime.Unit.US) - self._initialization_time
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
        pass

    async def RegisterEnvironmentReady(self, request, context):
        pass

    async def RegisterWorker(self, request, context):
        pass

    async def GetPlacements(self, request, context):
        pass

    async def NotifyTaskCompletion(self, request, context):
        pass


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
