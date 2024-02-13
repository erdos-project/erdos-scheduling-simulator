import os
import sys
import time
from concurrent import futures
from typing import Mapping
from urllib.parse import urlparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import erdos_scheduler_pb2
import erdos_scheduler_pb2_grpc
import grpc
from absl import app, flags

from utils import EventTime, setup_logging
from workers import Worker, WorkerPool
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Resource,
    Resources,
    Task,
    WorkProfile,
)

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

        # The simulator types maintained by the Servicer.
        self._worker_pool = None
        self._drivers: Mapping[str, Task] = {}

        # Application (TaskGraph) information maintained by the Servicer.
        self._all_task_graphs = {}

        super().__init__()

    def RegisterFramework(self, request, context):
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

        # Return the response.
        return erdos_scheduler_pb2.RegisterFrameworkResponse(
            success=True,
            message=f"{framework_name} at {self._master_uri} registered successfully!",
        )

    def RegisterDriver(self, request, context):
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
                worker_name="",
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
            resource_vector={Resource(name="Slot_CPU", _id="any"): request.cores}
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
                            # fixed runtime. We set it to invalid to indicate this.
                            runtime=EventTime.invalid(),
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
                    worker.place_task(driver, execution_strategy)
                    return erdos_scheduler_pb2.RegisterDriverResponse(
                        success=True,
                        message=f"Driver {request.id} registered successfully!",
                        worker_name=worker.name,
                        worker_id=worker.name,
                    )

        if not placement_found:
            return erdos_scheduler_pb2.RegisterDriverResponse(
                success=False,
                message=f"No Worker can accomodate the driver {request.id} yet.",
                worker_name="",
                worker_id="",
            )

    def RegisterTaskGraph(self, request, context):
        """Registers a new TaskGraph with the backend scheduler.
        This is the entry point for a new application of Spark to register
        itself with the backend scheduler, and is intended as an EHLO.
        """
        app_id = request.id
        app_name = request.name
        received_ts = time.time()
        if app_id in self._all_task_graphs:
            self._logger.warning(
                "The application with ID %s and name %s was already registered.",
                app_id,
                app_name,
            )
            return erdos_scheduler_pb2.RegisterTaskGraphResponse(
                success=False,
                message=f"Application ID {app_id} with name {app_name} "
                f"already registered!",
            )

        # Setup a new TaskGraph (application).
        self._logger.info(
            "Registering application ID %s with name %s at received_ts %s",
            app_id,
            app_name,
            received_ts,
        )

        # Setup application information for servicer.
        new_application = {
            "app_id": app_id,
            "app_name": app_name,
            "received_ts": received_ts,
        }
        self._all_task_graphs[app_id] = new_application

        # Return the response.
        return erdos_scheduler_pb2.RegisterTaskGraphResponse(
            success=True,
            message=f"Application ID {app_id} with name "
            f"{app_name} registered successfully!",
        )

    def DeregisterFramework(self, request, context):
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

    def RegisterWorker(self, request, context):
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
        worker_resources = Resources(resource_vector={cpu_resource: request.cores})
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

        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True, message=f"Worker {request.name} registered successfully!"
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
