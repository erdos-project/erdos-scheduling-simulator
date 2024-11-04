import asyncio
from concurrent import futures

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
        pass

    async def RegisterFramework(self, request, context):
        pass

    async def DeregisterFramework(self, request, context):
        pass

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
    erdos_scheduler_pb2_grpc.add_SchedulerServiceServicer_to_server(
        Servicer(), server
    )
    server.add_insecure_port(f"[::]:{FLAGS.port}")

    try:
        loop.run_until_complete(serve(server))
    except KeyboardInterrupt:
        print("Terminated ERDOS RPC Service")
    finally:
        loop.close()


if __name__ == "__main__":
    app.run(main)
