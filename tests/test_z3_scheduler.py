import pytest

from schedulers import Z3Scheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph, Workload


def test_z3_scheduling_success_basic():
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
        timestamp=0,
    )
    task_graph = TaskGraph(name="TestTaskGraph", tasks={camera_task_1: []})
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())

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
        runtime=EventTime.zero(),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was expected to be placed."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_placement.placement_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."


@pytest.mark.skip(reason="Succeeds locally but fails on CI.")
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
    task_graph = TaskGraph(name="TestTaskGraph", tasks={camera_task_1: []})
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())

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
        runtime=EventTime.zero(),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert not camera_task_placement.is_placed(), "Incorrect WorkerPoolID retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime.zero(),
        lookahead=EventTime.zero(),
        enforce_deadlines=False,
    )
    placements = scheduler.schedule(EventTime.zero(), task_graph, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_placement.placement_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."


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
    task_graph = TaskGraph(
        name="TestTaskGraph", tasks={camera_task_1: [perception_task_1]}
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    perception_task_1.release(EventTime.zero())

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
        runtime=EventTime.zero(),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_placement.placement_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."

    perception_task_placement = placements.get_placement(perception_task_1)
    assert (
        perception_task_placement is not None
    ), "The task was not found in placements."
    assert (
        perception_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert perception_task_placement.placement_time == EventTime(
        5, EventTime.Unit.US
    ), "Incorrect start time retrieved."


def test_z3_skip_tasks_under_enforce_deadlines():
    """Test that if some task deadlines cannot be met, the remainder are scheduled
    instead of not placing any task in the system."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1", timestamp=0, runtime=5, deadline=5
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
        runtime=8,
        deadline=10,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph", tasks={camera_task_1: [perception_task_1]}
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    perception_task_1.release(EventTime.zero())

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
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was expected in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_placement.placement_time == EventTime(
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
    task_graph = TaskGraph(
        name="TestTaskGraph", tasks={camera_task_1: [], camera_task_2: []}
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

    # Create the workers.
    worker_1 = Worker(name="Worker_1", resources=Resources({Resource(name="CPU"): 1}))
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = Z3Scheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was expected in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placement.placement_time == EventTime.zero()
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was expected in placements."
    assert (
        camera_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_2_placement.placement_time == EventTime(
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
        name="TestTaskGraph",
        tasks={camera_task_1: [], camera_task_2: [perception_task_2]},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())
    perception_task_2.release(EventTime.zero())

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
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."
    assert (
        camera_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_1_placement.placement_time == EventTime.zero()
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert (
        camera_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_2_placement.placement_time == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect start time retrieved."

    perception_task_2_placement = placements.get_placement(perception_task_2)
    assert (
        perception_task_2_placement is not None
    ), "The task was not found in placements."
    assert (
        perception_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert perception_task_2_placement.placement_time == EventTime(
        11, EventTime.Unit.US
    ), "Incorrect placement for perception_task_2."


@pytest.mark.skip(reason="Changing the optimization objective fails the test.")
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
        name="TestTaskGraph",
        tasks={camera_task_1: [], camera_task_2: [perception_task_2]},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())
    perception_task_2.release(EventTime.zero())

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
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    runtime, placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert (
        camera_task_1,
        worker_pool_1.id,
        EventTime.zero(),
    ) in placements, "Incorrect placement for camera_task_1."
    assert (
        camera_task_2,
        None,
        None,
    ) in placements, "Incorrect placement for camera_task_2."
    assert (
        perception_task_2,
        None,
        None,
    ) in placements, "Incorrect placement for camera_task_2."


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

    task_graph = TaskGraph(
        name="TestTaskGraph", tasks={camera_task_1: [], camera_task_2: []}
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

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
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert not (
        placements.get_placement(camera_task_1).is_placed()
        and placements.get_placement(camera_task_2).is_placed()
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

    task_graph = TaskGraph(
        name="TestTaskGraph", tasks={camera_task_1: [], camera_task_2: []}
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

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
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."
    assert (
        camera_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_1_placement.placement_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert not camera_task_2_placement.is_placed(), "Incorrect WorkerPoolID retrieved."


@pytest.mark.skip(reason="Changing the optimization objective fails the test.")
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
        name="TestTaskGraph",
        tasks={camera_task_1: [perception_task_1], camera_task_2: []},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

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
    runtime, placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."
    assert (
        camera_task_1,
        worker_pool_1.id,
        EventTime.zero(),
    ) in placements, "Incorrect placement for camera_task_1."
    assert (
        camera_task_2,
        worker_pool_1.id,
        EventTime(20, EventTime.Unit.US),
    ) in placements, "Incorrect placement for camera_task_2."
    assert (
        perception_task_1,
        worker_pool_1.id,
        EventTime(11, EventTime.Unit.US),
    ) in placements, "Incorrect placement for camera_task_2."


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
        name="TestTaskGraph",
        tasks={camera_task_1: [perception_task_1], camera_task_2: []},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())
    perception_task_1.release(EventTime.zero())

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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."
    assert (
        camera_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_1_placement.placement_time == EventTime.zero()
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert (
        camera_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert camera_task_2_placement.placement_time == EventTime(
        12, EventTime.Unit.US
    ), "Incorrect start time retrieved."

    perception_task_1_placement = placements.get_placement(perception_task_1)
    assert (
        perception_task_1_placement is not None
    ), "The task was not found in placements."
    assert (
        perception_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert perception_task_1_placement.placement_time == EventTime(
        10, EventTime.Unit.US
    ), "Incorrect placement for perception_task_2."
