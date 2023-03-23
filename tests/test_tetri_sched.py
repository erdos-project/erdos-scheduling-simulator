from schedulers import TetriSched
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph, Workload


def test_tetri_sched_success_basic():
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
        timestamp=0,
        deadline=1100,
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
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime.zero(),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was expected to be placed."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placement.placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."


def test_ilp_scheduling_multiple_tasks_different_resources():
    """Test that TetriSched schedules multiple tasks requiring different
    resources correctly."""
    # Create the tasks and the graph.
    cpu_task = create_default_task(
        name="CPU_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        deadline=200,
    )
    cpu_gpu_task = create_default_task(
        name="CPU_GPU_task",
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        deadline=50,
    )
    task_graph = TaskGraph(name="TestTaskGraph", tasks={cpu_task: [], cpu_gpu_task: []})
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    cpu_task.release(EventTime.zero())
    cpu_gpu_task.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])

    worker_2 = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_2 = WorkerPool(name="WorkerPool_2", workers=[worker_2])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1, worker_pool_2])

    # Create the scheduler.
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime.zero(),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."

    cpu_task_placement = placements.get_placement(cpu_task)
    assert cpu_task_placement is not None, "The task was expected to be placed."
    assert (
        cpu_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        cpu_task_placement.placement_time + cpu_task.remaining_time <= cpu_task.deadline
    ), "Incorrect start time retrieved."

    cpu_gpu_task_placement = placements.get_placement(cpu_gpu_task)
    assert cpu_gpu_task_placement is not None, "The task was expected to be placed."
    assert (
        cpu_gpu_task_placement.worker_pool_id == worker_pool_2.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        cpu_gpu_task_placement.placement_time + cpu_gpu_task.remaining_time
        <= cpu_task.deadline
    ), "Incorrect start time retrieved."


def test_tetri_sched_limited_resources():
    """Test that TetriSched recognizes that the Workload is not schedulable."""
    # Create the tasks and the graph.
    task_one = create_default_task(name="Task 1", deadline=200, runtime=100)
    task_two = create_default_task(name="Task 2", deadline=220, runtime=150)
    task_graph = TaskGraph(name="TestTaskGraph", tasks={task_one: [], task_two: []})
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})

    # Release the tasks.
    release_time = EventTime(50, EventTime.Unit.US)
    task_one.release(release_time)
    task_two.release(release_time)

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime.zero(),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(release_time, workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."
    task_one_placement = placements.get_placement(task_one)
    task_two_placement = placements.get_placement(task_two)
    assert not (
        task_one_placement.is_placed() and task_two_placement.is_placed()
    ), "Only one task should be placed."


def test_ilp_scheduling_deadline_enforcement():
    """Tests that ILP tries to schedule the task under soft deadline enforcement."""
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
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime.zero(),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert not camera_task_placement.is_placed(), "Incorrect WorkerPoolID retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime.zero(),
        enforce_deadlines=False,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placement.placement_time <= camera_task_1.deadline
    ), "Incorrect start time retrieved."


def test_ilp_delays_scheduling_under_constrained_resources():
    """Tests that if the resources are constrained, ILP delays the execution of some
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
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was expected in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placement.placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was expected in placements."
    assert (
        camera_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_2_placement.placement_time + camera_task_2.remaining_time
        <= camera_task_2.deadline
    ), "Incorrect start time retrieved."


def test_ilp_respects_worker_resource_constraints():
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
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert not (
        placements.get_placement(camera_task_1).is_placed()
        and placements.get_placement(camera_task_2).is_placed()
    ), "One of the tasks should not be placed."


def test_ilp_does_not_schedule_across_workers():
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
    scheduler = TetriSched(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        enforce_deadlines=True,
        goal="tetri_sched_naive",
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert not camera_task_2_placement.is_placed(), "Incorrect WorkerPoolID retrieved."
