import asyncio
import os
import sys
import time
from collections import defaultdict
from concurrent import futures
from typing import Mapping, Sequence
from urllib.parse import urlparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import erdos_scheduler_pb2
import erdos_scheduler_pb2_grpc
import grpc
from absl import app, flags

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
flags.DEFINE_string("log_file", None, "Path to the log file.", short_name="log")
flags.DEFINE_string("log_level", "debug", "The level to log.")
flags.DEFINE_integer(
    "initial_executors",
    10,
    "The initial number of executors that are requested by each application.",
)


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
        self._worker_pools = None
        self._drivers: Mapping[str, Task] = {}
        self._workload = None

        # Scheduler information maintained by the servicer.
        self._scheduler_running_lock = asyncio.Lock()
        self._scheduler_running = False
        self._rerun_scheduler = False

        # Placement information maintained by the servicer.
        # The placements map the application IDs to the Placement retrieved from the
        # scheduler. The placements are automatically clipped at the time of informing
        # the framework of applying them to the executors.
        # NOTE (Sukrit): This must always be sorted by the Placement time.
        self._placements: Mapping[str, Sequence[Placement]] = defaultdict(list)

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

        current_time = EventTime(int(time.time()), EventTime.Unit.S)
        self._logger.info(
            "Starting a scheduling cycle with %s TaskGraphs and %s Workers at %s.",
            len(self._workload.task_graphs),
            len(self._worker_pool.workers),
            current_time,
        )

        # TODO (Sukrit): Change this to a better implementation.
        # Let's do some simple scheduling for now, that gives a fixed number of
        # executors to all the available applications in intervals of 10 seconds.
        if len(self._workload.task_graphs) > 0:
            tasks = self._workload.get_schedulable_tasks(
                current_time, worker_pools=self._worker_pools
            )
            self._logger.info(
                "Found %s tasks that can be scheduled at %s: %s",
                len(tasks),
                current_time,
                [task.unique_name for task in tasks],
            )
            if len(tasks) > 0:
                task = tasks[0]
                strategy = task.available_execution_strategies.get_fastest_strategy()
                placement = Placement(
                    type=Placement.PlacementType.PLACE_TASK,
                    computation=tasks[0],
                    placement_time=EventTime(int(time.time()) + 5, EventTime.Unit.S),
                    worker_pool_id=self._worker_pool.id,
                    worker_id=self._worker_pool.workers[0].name,
                    strategy=strategy,
                )
                self._placements[task.task_graph].append(placement)
                task.schedule(
                    time=placement.placement_time,
                    placement=placement,
                )

        self._logger.info("Finished a scheduling cycle.")

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
                    placement_time = EventTime(request.timestamp, EventTime.Unit.S)
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
        completion_time = EventTime(request.timestamp, EventTime.Unit.S)
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

        # Construct all the Tasks for the TaskGraph.
        task_ids_to_task: Mapping[int, Task] = {}
        for task_dependency in request.dependencies:
            framework_task = task_dependency.key
            task_ids_to_task[framework_task.id] = Task(
                name=framework_task.name,
                task_graph=request.id,
                job=Job(
                    name=framework_task.name,
                    profile=WorkProfile(
                        name=f"WorkProfile_{framework_task.name}",
                        execution_strategies=ExecutionStrategies(
                            [
                                # TODO (Sukrit): Find the actual resource requirements
                                # for the particular TaskGraph, along with the expected
                                # runtime and set it here.
                                ExecutionStrategy(
                                    resources=Resources(
                                        resource_vector={
                                            Resource(name="Slot_CPU", _id="any"): 1
                                        }
                                    ),
                                    batch_size=1,
                                    runtime=EventTime(20, EventTime.Unit.US),
                                )
                            ]
                        ),
                    ),
                ),
                deadline=EventTime(request.deadline, EventTime.Unit.S),
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
            source_task.release(EventTime(request.timestamp, EventTime.Unit.S))

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

        # Run the scheduler since the Resource set has changed, and new task graphs
        # may become eligible to run.
        await self.run_scheduler()

        return erdos_scheduler_pb2.RegisterWorkerResponse(
            success=True, message=f"Worker {request.name} registered successfully!"
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

        # Mark the Task as completed.
        matched_task.update_remaining_time(EventTime.zero())
        matched_task.finish(EventTime(request.timestamp, EventTime.Unit.S))

        # Run the scheduler since the Workload has changed.
        await self.run_scheduler()

        return erdos_scheduler_pb2.NotifyTaskCompletionResponse(
            success=True,
            message=f"Task with ID {request.task_id} completed successfully!",
        )

    async def GetPlacements(self, request, context):
        """Retrieves the placements applicable at the specified time."""
        request_timestamp = EventTime(request.timestamp, EventTime.Unit.S)
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
        self._logger.info(
            "Currently %s placements, clipping at %s.", len(placements), clip_at
        )
        self._placements[request.id] = self._placements[request.id][clip_at + 1 :]
        self._logger.info(
            "Clipped placements length: %s", len(self._placements[request.id])
        )
        self._logger.info(
            "Constructed %s placements at time %s.", len(placements), request.timestamp
        )
        return erdos_scheduler_pb2.GetPlacementsResponse(
            success=True,
            placements=placements,
            message=f"Constructed {len(placements)} "
            f"placements at time {request.timestamp}.",
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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(serve())
    loop.close()


if __name__ == "__main__":
    app.run(main)
