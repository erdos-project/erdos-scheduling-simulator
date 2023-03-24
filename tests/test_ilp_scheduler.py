import os

import pytest

from schedulers import ILPScheduler, TetriSchedCPLEXScheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph, Workload


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_success_basic(scheduler):
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


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_multiple_tasks_different_resources(scheduler):
    """Test that the ILPScheduler schedules multiple tasks requiring different
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


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_scheduler_limited_resources(scheduler):
    """Test that the ILPScheduler recognizes that the Workload is not schedulable."""
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
    placements = scheduler.schedule(release_time, workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."
    task_one_placement = placements.get_placement(task_one)
    task_two_placement = placements.get_placement(task_two)
    assert not (
        task_one_placement.is_placed() and task_two_placement.is_placed()
    ), "Only one task should be placed."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
            goal="max_slack",
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_deadline_enforcement(scheduler):
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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert not camera_task_placement.is_placed(), "Incorrect WorkerPoolID retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler._enforce_deadlines = False
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


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_dependency(scheduler):
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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    assert camera_task_placement is not None, "The task was not found in placements."
    assert (
        camera_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placement.placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    perception_task_placement = placements.get_placement(perception_task_1)
    assert (
        perception_task_placement is not None
    ), "The task was not found in placements."
    assert (
        perception_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        perception_task_placement.placement_time + perception_task_1.remaining_time
        <= perception_task_1.deadline
    ), "Incorrect start time retrieved."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
            release_taskgraphs=True,
        ),
    ],
    ids=["ILP"],
)
def test_ilp_skip_taskgraphs_under_enforce_deadlines(scheduler):
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

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2, Resource(name="GPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_placement = placements.get_placement(camera_task_1)
    perception_task_placement = placements.get_placement(perception_task_1)
    assert (
        camera_task_placement is not None and perception_task_placement is not None
    ), "Placement information for the tasks was not provided."
    assert (
        not camera_task_placement.is_placed()
        and not perception_task_placement.is_placed()
    ), "The tasks were not expected to be placed."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_delays_scheduling_under_constrained_resources(scheduler):
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


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
    ],
    ids=["ILP"],
)
def test_ilp_respects_dependencies_under_delayed_scheduling(scheduler):
    """Tests that if the resources are constrained, ILP still respects dependencies
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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."
    assert (
        camera_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_1_placement.placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert (
        camera_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_2_placement.placement_time + camera_task_2.remaining_time
        <= camera_task_2.deadline
    ), "Incorrect start time retrieved."

    perception_task_2_placement = placements.get_placement(perception_task_2)
    assert (
        perception_task_2_placement is not None
    ), "The task was not found in placements."
    assert (
        perception_task_2_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        perception_task_2_placement.placement_time + perception_task_2.remaining_time
        <= perception_task_2.deadline
    ), "Incorrect placement for perception_task_2."


def test_ilp_respects_dependencies_under_constrained_resources():
    """Tests that if upstream tasks cannot be scheduled due to resource constraints,
    downstream tasks are also not scheduled."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        timestamp=0,
        runtime=5,
        deadline=10,
        task_graph_name="TestTaskGraph1",
    )
    task_graph_1 = TaskGraph(
        name="TestTaskGraph1",
        tasks={camera_task_1: []},
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
        task_graph_name="TestTaskGraph2",
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
        task_graph_name="TestTaskGraph2",
    )
    task_graph_2 = TaskGraph(
        name="TestTaskGraph2",
        tasks={camera_task_2: [perception_task_2]},
    )
    workload = Workload.from_task_graphs(
        {"TestTaskGraph1": task_graph_1, "TestTaskGraph2": task_graph_2}
    )
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
    scheduler = ILPScheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "No placement found for camera_task_1."
    assert (
        camera_task_1_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect worker placement for camera_task_1"
    assert (
        camera_task_1_placement.placement_time + camera_task_1.runtime
        <= camera_task_1.deadline
    ), "Invalid start time for camera_task_1."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "No placement found for camera_task_2."
    assert (
        not camera_task_2_placement.is_placed()
    ), "camera_task_2 should not be placed."

    perception_task_2_placement = placements.get_placement(perception_task_2)
    assert (
        not perception_task_2_placement.is_placed()
    ), "No placement found for perception_task_2."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_respects_worker_resource_constraints(scheduler):
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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."
    assert not (
        placements.get_placement(camera_task_1).is_placed()
        and placements.get_placement(camera_task_2).is_placed()
    ), "One of the tasks should not be placed."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS"),
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_does_not_schedule_across_workers(scheduler):
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
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "The task was not found in placements."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "The task was not found in placements."
    assert not camera_task_2_placement.is_placed(), "Incorrect WorkerPoolID retrieved."


def test_ilp_not_work_conserving():
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
    scheduler = ILPScheduler(
        preemptive=False,
        runtime=EventTime(-1, EventTime.Unit.US),
        lookahead=EventTime(50, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placement = placements.get_placement(camera_task_1)
    assert camera_task_1_placement is not None, "Placement for camera_task_1 not found."
    assert camera_task_1_placement.placement_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect placement time for camera_task_1."

    perception_task_1_placement = placements.get_placement(perception_task_1)
    assert (
        perception_task_1_placement is not None
    ), "Placement for perception_task_1 not found."
    assert perception_task_1_placement.placement_time == EventTime(
        12, EventTime.Unit.US
    ), "Incorrect placement time for perception_task_1."

    camera_task_2_placement = placements.get_placement(camera_task_2)
    assert camera_task_2_placement is not None, "Placement for camera_task_2 not found."
    assert camera_task_2_placement.placement_time == EventTime(
        21, EventTime.Unit.US
    ), "Incorrect placement time for camera_task_2."
