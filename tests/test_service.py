import re
import time
import subprocess

import pytest
import grpc
from rpc import erdos_scheduler_pb2
from rpc import erdos_scheduler_pb2_grpc


@pytest.fixture(scope="module", autouse=True)
def service():
    process = subprocess.Popen(["python", "-m", "rpc.service"])
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

    # Register an incorrect TaskGraph
    request = erdos_scheduler_pb2.RegisterTaskGraphRequest(
        id="task-graph",
        name="TPCH Query 4 50 50",
        timestamp=1234567890,
        dependencies=[
            {"key": {"id": 0, "name": "stage 0"}, "children_ids": [1, 2]},
        ],
    )
    response = stub.RegisterTaskGraph(request)
    assert not response.success and re.search(
        r"Failed to load TPCH query 4. Exception: Structure of dependencies provided for query number 4 does not match that of canonical dependencies",
        response.message,
    )

    # Register a correct TaskGraph
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
            r"Registered task graph 'Q4\[task-graph-0\]@1' successfully",
            response.message,
        )
        and response.num_executors == 10
    )

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
    assert False

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
