from schedulers import Z3Scheduler
from tests.utils import create_default_task
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph


def test_z3_scheduling_success_basic():
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", job=Job(name="Camera_1", runtime=1000), timestamp=0
    )
    task_graph = TaskGraph(tasks={camera_task_1: []})
    camera_task_1.release(0)

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False, runtime=0, lookahead=0, enforce_deadlines=True
    )
    runtime, placements = scheduler.schedule(0, task_graph, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == 0, "Incorrect start time retrieved."


def test_z3_scheduling_deadline_enforcement():
    """Tests that Z3 tries to schedule the task under soft deadline enforcement."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1", runtime=1000),
        timestamp=0,
        runtime=5,
        deadline=2,
    )
    task_graph = TaskGraph(tasks={camera_task_1: []})
    camera_task_1.release(0)

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the enforce deadlines scheduler.
    scheduler = Z3Scheduler(
        preemptive=False, runtime=0, lookahead=0, enforce_deadlines=True
    )
    runtime, placements = scheduler.schedule(0, task_graph, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == None, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == None, "Incorrect start time retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler = Z3Scheduler(
        preemptive=False, runtime=0, lookahead=0, enforce_deadlines=False
    )
    runtime, placements = scheduler.schedule(0, task_graph, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == 0, "Incorrect start time retrieved."


def test_z3_scheduling_dependency():
    """Ensure that the dependencies are correctly scheduled."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(name="Camera_1", timestamp=0, runtime=5)
    perception_task_1 = create_default_task(
        name="Perception_1",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=8,
        deadline=20,
    )
    task_graph = TaskGraph(tasks={camera_task_1: [perception_task_1]})
    camera_task_1.release(0)
    perception_task_1.release(0)

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2, Resource(name="GPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False, runtime=0, lookahead=0, enforce_deadlines=True
    )
    runtime, placements = scheduler.schedule(0, task_graph, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == 0, "Incorrect start time retrieved."
    assert (
        placements[1][0] == perception_task_1
    ), "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == 5, "Incorrect start time retrieved."
