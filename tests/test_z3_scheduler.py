from schedulers import Z3Scheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph


def test_z3_scheduling_success_basic():
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    task_graph = TaskGraph(tasks={camera_task_1: []})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(0, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_scheduling_deadline_enforcement():
    """Tests that Z3 tries to schedule the task under soft deadline enforcement."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        runtime=5,
        deadline=2,
    )
    task_graph = TaskGraph(tasks={camera_task_1: []})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the enforce deadlines scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(0, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] is None, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] is None, "Incorrect start time retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(0, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=False,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), f"Incorrect start time retrieved: {placements[0][2]}."


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
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    perception_task_1.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2, Resource(name="GPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(0, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )

    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert (
        placements[1][0] == perception_task_1
    ), "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == EventTime(
        5, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_skip_tasks_under_enforce_deadlines():
    """Test that if some task deadlines cannot be met, the remainder are scheduled
    instead of not placing any task in the system."""
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
        deadline=10,
    )
    task_graph = TaskGraph(tasks={camera_task_1: [perception_task_1]})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    perception_task_1.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2, Resource(name="GPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_delays_scheduling_under_constrained_resources():
    """Tests that if the resources are constrained, z3 delays the execution of some
    tasks instead of skipping their execution."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=5, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2", timestamp=0, runtime=5, deadline=20
    )
    task_graph = TaskGraph(tasks={camera_task_1: [], camera_task_2: []})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(name="Worker_1", resources=Resources({Resource(name="CPU"): 1}))
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[1][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_respects_dependencies_under_delayed_scheduling():
    """Tests that if the resources are constrained, z3 still respects dependencies
    amongst the tasks."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=5, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2", timestamp=0, runtime=5, deadline=20
    )
    perception_task_2 = create_default_task(
        name="Perception_2",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=8,
        deadline=30,
    )
    task_graph = TaskGraph(
        tasks={camera_task_1: [], camera_task_2: [perception_task_2]}
    )
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))
    perception_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[1][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert (
        placements[2][0] == perception_task_2
    ), "Incorrect task retrieved for placement."
    assert placements[2][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[2][2] == EventTime(
        11, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_respects_dependencies_under_constrained_resources():
    """Tests that if upstream tasks cannot be scheduled due to resource constraints,
    downstream tasks are also not scheduled."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=5, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 3,
            }
        ),
        runtime=5,
        deadline=20,
    )
    perception_task_2 = create_default_task(
        name="Perception_2",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=8,
        deadline=30,
    )
    task_graph = TaskGraph(
        tasks={camera_task_1: [], camera_task_2: [perception_task_2]}
    )
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))
    perception_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[1][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[1][1] is None, "Incorrect WorkerPoolID retrieved."
    assert (
        placements[2][0] == perception_task_2
    ), "Incorrect task retrieved for placement."
    assert placements[2][1] is None, "Incorrect WorkerPoolID retrieved."


def test_z3_respects_worker_resource_constraints():
    """Tests that the scheduler respects the maximum resources in the worker."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=10, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=10,
        deadline=10,
    )

    task_graph = TaskGraph(tasks={camera_task_1: [], camera_task_2: []})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert (
        placements[0][1] is None or placements[1][1] is None
    ), "One of the tasks should not be placed."


def test_z3_does_not_schedule_across_workers():
    """Tests that the scheduler restricts the allocation to individual workers."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=10, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=10,
        deadline=10,
    )

    task_graph = TaskGraph(tasks={camera_task_1: [], camera_task_2: []})
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_2 = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1, worker_2])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(0, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[1][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[1][1] is None, "Incorrect WorkerPoolID retrieved."


def test_z3_not_work_conserving():
    """Tests that the scheduler restricts the allocation to individual workers."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=10, deadline=15
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        timestamp=0,
        runtime=10,
        deadline=50,
    )
    perception_task_1 = create_default_task(
        name="Perception_1",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        release_time=11,
        runtime=8,
        deadline=20,
    )

    task_graph = TaskGraph(
        tasks={camera_task_1: [perception_task_1], camera_task_2: []}
    )
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(50, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[1][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == EventTime(
        20, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert (
        placements[2][0] == perception_task_1
    ), "Incorrect task retrieved for placement."
    assert placements[2][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[2][2] == EventTime(
        11, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_minimize_deadline_misses():
    """Test that the scheduler can minimize missed deadlines."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=10, deadline=10
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        timestamp=0,
        runtime=10,
        deadline=15,
    )
    perception_task_1 = create_default_task(
        name="Perception_1",
        timestamp=0,
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        runtime=1,
        deadline=20,
    )

    task_graph = TaskGraph(
        tasks={camera_task_1: [perception_task_1], camera_task_2: []}
    )
    camera_task_1.release(EventTime(0, EventTime.Unit.US))
    camera_task_2.release(EventTime(0, EventTime.Unit.US))
    perception_task_1.release(EventTime(0, EventTime.Unit.US))

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(50, EventTime.Unit.US),
        enforce_deadlines=False,
    )
    runtime, placements = scheduler.schedule(
        EventTime(0, EventTime.Unit.US), task_graph, worker_pools
    )
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert placements[0][0] == camera_task_1, "Incorrect task retrieved for placement."
    assert placements[0][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[0][2] == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert (
        placements[1][0] == perception_task_1
    ), "Incorrect task retrieved for placement."
    assert placements[1][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[1][2] == EventTime(
        10, EventTime.Unit.US
    ), "Incorrect start time retrieved."
    assert placements[2][0] == camera_task_2, "Incorrect task retrieved for placement."
    assert placements[2][1] == worker_pool_1.id, "Incorrect WorkerPoolID retrieved."
    assert placements[2][2] == EventTime(
        12, EventTime.Unit.US
    ), "Incorrect start time retrieved."
