import re
import time
import subprocess

import pytest
import grpc
from rpc import erdos_scheduler_pb2
from rpc import erdos_scheduler_pb2_grpc


@pytest.fixture(scope="module", autouse=True)
def service():
    process = subprocess.Popen(["python", "-m", "rpc.service", "--enforce_deadlines"])
    channel = grpc.insecure_channel("localhost:50051")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        yield process
    finally:
        channel.close()
        process.kill()


def test_service():
    channel = grpc.insecure_channel("localhost:50051")
    stub = erdos_scheduler_pb2_grpc.SchedulerServiceStub(channel)

    # Register a framework
    request = erdos_scheduler_pb2.RegisterFrameworkRequest(
        name="test_framework", uri="http://localhost/test", timestamp=1234567890
    )
    response = stub.RegisterFramework(request)
    assert response.success and re.search(
        r"Registered the framework 'test_framework' with URI http://localhost/test at UNIX time",
        response.message,
    )

    # Register a worker
    request = erdos_scheduler_pb2.RegisterWorkerRequest(
        name="test_worker",
        id="1234",
        cores=100,
        memory=1024,
    )
    response = stub.RegisterWorker(request)
    assert response.success and re.search(
        r"Registered worker \(id=1234, name=test_worker\)", response.message
    )

    # Try to fetch placements for an unregistered task graph
    # Get placements for the task, should be empty
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-0",
    )
    response = stub.GetPlacements(request)
    assert not response.success and re.search(
        r"Task graph with id \'task-graph-0\' is not registered or does not exist",
        response.message,
    )

    # TODO: move to environment ready
    # Register an incorrect TaskGraph
    # request = erdos_scheduler_pb2.RegisterTaskGraphRequest(
    #     id="task-graph",
    #     name="TPCH Query 4 50 50",
    #     timestamp=1234567890,
    #     dependencies=[
    #         {"key": {"id": 0, "name": "stage 0"}, "children_ids": [1, 2]},
    #     ],
    # )
    # response = stub.RegisterTaskGraph(request)
    # assert not response.success and re.search(
    #     r"Failed to load TPCH query 4. Exception: Structure of dependencies provided for query number 4 does not match that of canonical dependencies",
    #     response.message,
    # )

    # Register the first (correct) TaskGraph, it will be able to run
    request = erdos_scheduler_pb2.RegisterTaskGraphRequest(
        id="task-graph-0",
        name="TPCH Query 4 50 50",
        timestamp=1234567890,
        dependencies=[
            {"key": {"id": 0, "name": "stage 0"}, "children_ids": [2]},
            {"key": {"id": 1, "name": "stage 1"}, "children_ids": [2]},
            {"key": {"id": 2, "name": "stage 2"}, "children_ids": [3]},
            {"key": {"id": 3, "name": "stage 3"}, "children_ids": [4]},
            {"key": {"id": 4, "name": "stage 4"}, "children_ids": []},
        ],
    )
    response = stub.RegisterTaskGraph(request)
    assert (
        response.success
        and re.search(
            r"Registered task graph 'task-graph-0' successfully",
            response.message,
        )
        and response.num_executors == 10
    )

    # Introduce a 2s delay in getting the env ready
    time.sleep(2)

    # Mark the environment as ready
    request = erdos_scheduler_pb2.RegisterEnvironmentReadyRequest(
        id="task-graph-0",
        num_executors=10,
        timestamp=1234567890,
    )
    response = stub.RegisterEnvironmentReady(request)
    assert response.success and re.search(
        r"Successfully marked environment as ready for task graph 'Q4\[task-graph-0\]@1'",
        response.message,
    )

    time.sleep(3)

    # Get placements for the task
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-0",
    )
    response = stub.GetPlacements(request)
    assert response.success
    actual_task_ids = set()
    for placement in response.placements:
        assert (
            placement.worker_id == "1234" and placement.application_id == "task-graph-0"
        )
        actual_task_ids.add(placement.task_id)
    assert actual_task_ids == {0, 1}

    # Wait for 3 seconds and trigger notify task completion for tasks 0 and 1
    time.sleep(3)

    request = erdos_scheduler_pb2.NotifyTaskCompletionRequest(
        application_id="task-graph-0", task_id=0, timestamp=1234567890
    )
    response = stub.NotifyTaskCompletion(request)
    assert response.success

    request = erdos_scheduler_pb2.NotifyTaskCompletionRequest(
        application_id="task-graph-0", task_id=1, timestamp=1234567890
    )
    response = stub.NotifyTaskCompletion(request)
    assert response.success

    # Wait for 20s to allow the service to execute task completion for fastest task
    time.sleep(20)

    # Attempt to incorrectly notify task completion for task 3, which hasnt started yet
    request = erdos_scheduler_pb2.NotifyTaskCompletionRequest(
        application_id="task-graph-0", task_id=3, timestamp=1234567890
    )
    response = stub.NotifyTaskCompletion(request)
    assert not response.success

    # Wait 2s to allow the service to process the incorrect task completion
    time.sleep(2)

    # Wait for 25s to allow the service to finish execution of task 0
    time.sleep(25)

    # This will unlock task 2, which should now be returned as a placement
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-0",
    )
    response = stub.GetPlacements(request)
    assert response.success
    actual_task_ids = set()
    for placement in response.placements:
        assert (
            placement.worker_id == "1234" and placement.application_id == "task-graph-0"
        )
        actual_task_ids.add(placement.task_id)
    assert actual_task_ids == {2}

    # Attempt to register the second TaskGraph, wont be able to run due to inadequate resources
    request = erdos_scheduler_pb2.RegisterTaskGraphRequest(
        id="task-graph-1",
        name="TPCH Query 4 50 200",
        timestamp=1234567890,
        dependencies=[
            {"key": {"id": 0, "name": "stage 0"}, "children_ids": [2]},
            {"key": {"id": 1, "name": "stage 1"}, "children_ids": [2]},
            {"key": {"id": 2, "name": "stage 2"}, "children_ids": [3]},
            {"key": {"id": 3, "name": "stage 3"}, "children_ids": [4]},
            {"key": {"id": 4, "name": "stage 4"}, "children_ids": []},
        ],
    )
    response = stub.RegisterTaskGraph(request)
    assert (
        not response.success
        and re.search(
            r"The worker Pool cannot accomodate the task graph 'task-graph-1'",
            response.message,
        )
        and response.num_executors == 0
    )
    
    # Register the third TaskGraph, will run but will get cancelled due to deadline miss
    request = erdos_scheduler_pb2.RegisterTaskGraphRequest(
        id="task-graph-2",
        name="TPCH Query 4 50 50",
        timestamp=1234567890,
        dependencies=[
            {"key": {"id": 0, "name": "stage 0"}, "children_ids": [2]},
            {"key": {"id": 1, "name": "stage 1"}, "children_ids": [2]},
            {"key": {"id": 2, "name": "stage 2"}, "children_ids": [3]},
            {"key": {"id": 3, "name": "stage 3"}, "children_ids": [4]},
            {"key": {"id": 4, "name": "stage 4"}, "children_ids": []},
        ],
    )
    response = stub.RegisterTaskGraph(request)
    assert (
        response.success
        and re.search(
            r"Registered task graph 'task-graph-2' successfully",
            response.message,
        )
        and response.num_executors == 10
    )

    # Introduce a 2s delay in getting the env ready
    time.sleep(2)

    # Mark the environment as ready
    request = erdos_scheduler_pb2.RegisterEnvironmentReadyRequest(
        id="task-graph-2",
        num_executors=10,
        timestamp=1234567890,
    )
    response = stub.RegisterEnvironmentReady(request)
    assert response.success and re.search(
        r"Successfully marked environment as ready for task graph 'Q4\[task-graph-2\]@1'",
        response.message,
    )

    # Wait for 10s to get the placements for the second task graph
    time.sleep(10)

    # Get placements for the taskgraph 3, one of first two root vertices should be placed since there are resources
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-2",
    )
    response = stub.GetPlacements(request)
    assert response.success
    actual_task_ids = set()
    for placement in response.placements:
        assert (
            placement.worker_id == "1234" and placement.application_id == "task-graph-2"
        )
        actual_task_ids.add(placement.task_id)
    assert actual_task_ids == {1}

    # Wait for 100 more seconds and request placements again
    time.sleep(100)

    # Notify task completion for task 2 in task graph 0 to trigger scheduler run again
    request = erdos_scheduler_pb2.NotifyTaskCompletionRequest(
        application_id="task-graph-0", task_id=2, timestamp=1234567890
    )
    response = stub.NotifyTaskCompletion(request)
    assert response.success

    # Wait for 2 seconds to allow scheduler to process task completion and run scheduler
    time.sleep(2)

    # Get placements for the task, entire taskgraph would be cancelled since deadline has passed.
    # Since one root vertex (1) is running, the other root vertex (0) will be cancelled first,
    # then the subsequent vertices.
    # NOTE: The service will wait until all running/ scheduled tasks complete and are removed 
    # from the workerpool before issuing a terminate=True for the taskgraph. Until then it will 
    # return current placements for a taskgraph (including those already sent by the service) 
    # and wait for running tasks to finish. Spark will ignore it. 
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-2",
    )
    response = stub.GetPlacements(request)
    assert response.success
    actual_task_ids = set()
    # Will return placement for task_id 1
    for placement in response.placements:
        if placement.task_id == 1:
            assert (
                placement.worker_id == "1234" 
                and placement.application_id == "task-graph-2" 
            )
        actual_task_ids.add(placement.task_id)
    assert actual_task_ids == {1}
    
    # Wait for 5s to issue notify task completion for task_id 1 in task-graph-2
    time.sleep(5)
    request = erdos_scheduler_pb2.NotifyTaskCompletionRequest(
        application_id="task-graph-2", task_id=1, timestamp=1234567890
    )
    response = stub.NotifyTaskCompletion(request)
    assert response.success
    
    # Wait for 5s to allow the simulator to process the event. 
    # Invoke get placements again for task-graph 2, it should return terminate=True now
    time.sleep(5)
    request = erdos_scheduler_pb2.GetPlacementsRequest(
        timestamp=1234567890,
        id="task-graph-2",
    )
    response = stub.GetPlacements(request)
    assert response.success
    actual_task_ids = set()
    # Will return placement for task_id 1
    for placement in response.placements:
        actual_task_ids.add(placement.task_id)
    assert len(actual_task_ids) == 0
    assert response.terminate == True

    # Deregister framework
    request = erdos_scheduler_pb2.DeregisterFrameworkRequest(
        name="test_framework", uri="http://localhost/test", timestamp=1234567890
    )
    response = stub.DeregisterFramework(request)
    assert response.success and re.search(
        r"Successfully deregistered the framework at http://localhost/test",
        response.message,
    )

    channel.close()
