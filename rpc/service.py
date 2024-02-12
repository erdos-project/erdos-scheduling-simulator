import os
import sys
from concurrent import futures

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import erdos_scheduler_pb2
import erdos_scheduler_pb2_grpc
import grpc
from absl import app, flags

from utils import setup_logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 50051, "Port to serve the ERDOS Scheduling RPC Server on.")
flags.DEFINE_integer(
    "max_workers", 10, "Maximum number of workers to use for the RPC server."
)
flags.DEFINE_string("log_file", None, "Path to the log file.", short_name="log")
flags.DEFINE_string("log_level", "debug", "The level to log.")


# Implement the service.
class SchedulerServiceServicer(erdos_scheduler_pb2_grpc.SchedulerServiceServicer):
    def __init__(self) -> None:
        """Initialize the service, and setup the logger."""
        # Values used by the Servicer.
        self._logger = setup_logging(name=FLAGS.log_file, log_level=FLAGS.log_level)
        self._initialized = False
        self._initialization_time = -1
        self._master_uri = None
        super().__init__()

    def RegisterFramework(self, request, context):
        """Registers a new framework with the backend scheduler.
        This is the entry point for a new instance of Spark / Flink to register
        itself with the backend scheduler, and is intended as an EHLO.
        """
        if self._initialized:
            return erdos_scheduler_pb2.RegisterFrameworkResponse(
                success=False,
                message=f"Framework already registered at "
                f"{self._initialization_time} at the address {self._master_uri}",
            )

        # Setup a new Framework instance.
        framework_name = request.name
        self._master_uri = request.uri
        self._initialization_time = request.timestamp
        print("Registering framework %s with URI %s", framework_name, self._master_uri)
        return erdos_scheduler_pb2.RegisterFrameworkResponse(
            success=True,
            message=f"{framework_name} at {self._master_uri} registered successfully!",
        )


def serve(args):
    """Serves the ERDOS Scheduling RPC Server."""
    # Initialize the server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=FLAGS.max_workers))
    erdos_scheduler_pb2_grpc.add_SchedulerServiceServicer_to_server(
        SchedulerServiceServicer(), server
    )

    # Start the server.
    server.add_insecure_port(f"[::]:{FLAGS.port}")
    server.start()
    print("Initialized ERDOS Scheduling RPC Server on port", FLAGS.port)
    server.wait_for_termination()


if __name__ == "__main__":
    app.run(serve)
