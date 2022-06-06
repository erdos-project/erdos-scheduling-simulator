from schedulers import Z3Scheduler
from tests.utils import create_default_task
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources, TaskGraph


def test_basic_z3_scheduling_success():
    """Test that a single task can be successfully scheduled with enough resources."""
    scheduler = Z3Scheduler(
        preemptive=False, runtime=0, lookahead=0, enforce_deadlines=True
    )
    camera_task_1 = create_default_task(name="Camera_1", timestamp=0)
    task_graph = TaskGraph(tasks={camera_task_1: []})
    camera_task_1.release(0)
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])
    runtime, placements = scheduler.schedule(0, task_graph, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == 0, "Incorrect start time retrieved."
