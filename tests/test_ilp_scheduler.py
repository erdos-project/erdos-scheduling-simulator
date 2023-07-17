import os
from copy import deepcopy

import pytest

from schedulers import ILPScheduler, TetriSchedCPLEXScheduler, TetriSchedGurobiScheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    BatchStrategy,
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Resource,
    Resources,
    TaskGraph,
    Workload,
    WorkProfile,
)


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_success_basic(scheduler):
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
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
    camera_task_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_placements) == 1, "The task was expected to be placed."
    assert (
        camera_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placements[0].placement_time + camera_task_1.remaining_time
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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

    cpu_task_placements = placements.get_placements(cpu_task)
    assert len(cpu_task_placements) == 1, "The task was expected to be placed."
    assert (
        cpu_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        cpu_task_placements[0].placement_time + cpu_task.remaining_time
        <= cpu_task.deadline
    ), "Incorrect start time retrieved."

    cpu_gpu_task_placements = placements.get_placements(cpu_gpu_task)
    assert len(cpu_gpu_task_placements) == 1, "The task was expected to be placed."
    assert (
        cpu_gpu_task_placements[0].worker_pool_id == worker_pool_2.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        cpu_gpu_task_placements[0].placement_time + cpu_gpu_task.remaining_time
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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
    task_one_placements = placements.get_placements(task_one)
    task_two_placements = placements.get_placements(task_two)
    assert not (
        task_one_placements[0].is_placed() and task_two_placements[0].is_placed()
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
)
def test_ilp_scheduling_deadline_enforcement(scheduler):
    """Tests that ILP tries to schedule the task under soft deadline enforcement."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
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
    camera_task_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_placements) == 1, "The task was not found in placements."
    assert not camera_task_placements[
        0
    ].is_placed(), "Incorrect WorkerPoolID retrieved."

    # Create the softly enforce deadlines scheduler.
    scheduler._enforce_deadlines = False
    assert (
        not scheduler.enforce_deadlines
    ), "The Scheduler is still enforcing deadlines."
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 1, "Incorrect length of placements retrieved."
    camera_task_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_placements) == 1, "The task was not found in placements."
    assert (
        camera_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placements[0].placement_time <= camera_task_1.deadline
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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

    camera_task_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_placements) == 1, "The task was not found in placements."
    assert camera_task_placements[0].is_placed(), "The Task was not placed."
    assert (
        camera_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placements[0].placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    perception_task_placements = placements.get_placements(perception_task_1)
    assert len(perception_task_placements) == 1, "The task was not found in placements."
    assert (
        perception_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        perception_task_placements[0].placement_time + perception_task_1.remaining_time
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

    camera_task_placements = placements.get_placements(camera_task_1)
    perception_task_placements = placements.get_placements(perception_task_1)
    assert (
        len(camera_task_placements) == 1 and len(perception_task_placements) == 1
    ), "Placement information for the tasks was not provided."
    assert (
        not camera_task_placements[0].is_placed()
        and not perception_task_placements[0].is_placed()
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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

    camera_task_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_placements) == 1, "The task was expected in placements."
    assert (
        camera_task_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_placements[0].placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "The task was expected in placements."
    assert (
        camera_task_2_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_2_placements[0].placement_time + camera_task_2.remaining_time
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

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "The task was not found in placements."
    assert (
        camera_task_1_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_1_placements[0].placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "The task was not found in placements."
    assert (
        camera_task_2_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        camera_task_2_placements[0].placement_time + camera_task_2.remaining_time
        <= camera_task_2.deadline
    ), "Incorrect start time retrieved."

    perception_task_2_placements = placements.get_placements(perception_task_2)
    assert (
        len(perception_task_2_placements) == 1
    ), "The task was not found in placements."
    assert (
        perception_task_2_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        perception_task_2_placements[0].placement_time
        + perception_task_2.remaining_time
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

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "No placement found for camera_task_1."
    assert (
        camera_task_1_placements[0].worker_pool_id == worker_pool_1.id
    ), "Incorrect worker placement for camera_task_1"
    assert (
        camera_task_1_placements[0].placement_time
        + camera_task_1_placements[0].execution_strategy.runtime
        <= camera_task_1.deadline
    ), "Invalid start time for camera_task_1."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "No placement found for camera_task_2."
    assert not camera_task_2_placements[
        0
    ].is_placed(), "camera_task_2 should not be placed."

    perception_task_2_placements = placements.get_placements(perception_task_2)
    assert not perception_task_2_placements[
        0
    ].is_placed(), "No placement found for perception_task_2."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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
        placements.get_placements(camera_task_1)[0].is_placed()
        and placements.get_placements(camera_task_2)[0].is_placed()
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
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
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

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "The task was not found in placements."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "The task was not found in placements."
    assert not camera_task_2_placements[
        0
    ].is_placed(), "Incorrect WorkerPoolID retrieved."


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

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "Placement for camera_task_1 not found."
    assert camera_task_1_placements[0].placement_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect placement time for camera_task_1."

    perception_task_1_placements = placements.get_placements(perception_task_1)
    assert (
        len(perception_task_1_placements) == 1
    ), "Placement for perception_task_1 not found."
    assert perception_task_1_placements[0].placement_time == EventTime(
        12, EventTime.Unit.US
    ), "Incorrect placement time for perception_task_1."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "Placement for camera_task_2 not found."
    assert camera_task_2_placements[0].placement_time == EventTime(
        21, EventTime.Unit.US
    ), "Incorrect placement time for camera_task_2."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
        ),
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedGurobi", "TetriSchedCPLEX"],
)
def test_ilp_fits_correct_strategies(scheduler):
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
        profile=WorkProfile(
            name="Camera_1_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=220,
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        job=Job(name="Camera_2"),
        profile=WorkProfile(
            name="Camera_2_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(150, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=220,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={camera_task_1: [], camera_task_2: []},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 15}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Run the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 2, "Incorrect length of placements retrieved."

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "Placement for camera_task_1 not found."
    assert camera_task_1_placements[
        0
    ].is_placed(), "Placement for camera_task_1 failed."
    assert (
        camera_task_1_placements[0].execution_strategy
        == camera_task_1.profile.execution_strategies[1]
    ), "Incorrect strategy for camera_task_1."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "Placement for camera_task_2 not found."
    assert camera_task_2_placements[
        0
    ].is_placed(), "Placement for camera_task_1 failed."
    assert (
        camera_task_2_placements[0].placement_time
        <= camera_task_2.deadline
        - camera_task_2_placements[0].execution_strategy.runtime
    ), "Incorrect placement time for camera_task_2."


@pytest.mark.parametrize(
    "scheduler",
    [
        TetriSchedGurobiScheduler(
            runtime=EventTime.zero(),
            batching=True,
            enforce_deadlines=True,
            time_discretization=EventTime(10, EventTime.Unit.US),
            log_to_file=True,
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
            release_taskgraphs=True,
        ),
    ],
    ids=["TetriSchedGurobi", "ILP"],
)
def test_ilp_fits_correct_strategies_graph(scheduler):
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
        profile=WorkProfile(
            name="Camera_1_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=380,
        task_graph_name="TestGraph1",
    )
    perception_task_1 = create_default_task(
        name="Perception_1",
        job=Job(name="Perception_1"),
        profile=WorkProfile(
            name="Perception_1_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=380,
        task_graph_name="TestGraph1",
    )
    task_graph_1 = TaskGraph(
        name="TestGraph1",
        tasks={camera_task_1: [perception_task_1], perception_task_1: []},
    )

    camera_task_2 = create_default_task(
        name="Camera_2",
        job=Job(name="Camera_2"),
        profile=WorkProfile(
            name="Camera_2_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=380,
        task_graph_name="TestGraph2",
    )
    perception_task_2 = create_default_task(
        name="Perception_2",
        job=Job(name="Perception_2"),
        profile=WorkProfile(
            name="Perception_2_Profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="CPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    ),
                ]
            ),
        ),
        deadline=380,
        task_graph_name="TestGraph2",
    )
    task_graph_2 = TaskGraph(
        name="TestGraph2",
        tasks={camera_task_2: [perception_task_2], perception_task_2: []},
    )

    workload = Workload.from_task_graphs(
        {"TestGraph1": task_graph_1, "TestGraph2": task_graph_2}
    )
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker1",
        resources=Resources(resource_vector={Resource(name="CPU"): 15}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Run the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 4, "Incorrect number of placements."

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "Incorrect number of placements."
    assert camera_task_1_placements[
        0
    ].is_placed(), "Placement for camera_task_1 failed."

    perception_task_1_placements = placements.get_placements(perception_task_1)
    assert len(perception_task_1_placements) == 1, "Incorrect number of placements."
    assert perception_task_1_placements[
        0
    ].is_placed(), "Placement for perception_task_1 failed."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "Incorrect number of placements."
    assert camera_task_2_placements[
        0
    ].is_placed(), "Placement for camera_task_2 failed."

    perception_task_2_placements = placements.get_placements(perception_task_2)
    assert len(perception_task_2_placements) == 1, "Incorrect number of placements."
    assert perception_task_2_placements[
        0
    ].is_placed(), "Placement for perception_task_2 failed."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
            batching=True,
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                batching=True,
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["ILP", "TetriSchedCPLEX"],
)
def test_ilp_batching(scheduler):
    """Test that the ILP-based scheduler batches tasks correctly."""
    work_profile = WorkProfile(
        name="Camera_Profile",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 10}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 12}
                    ),
                    batch_size=2,
                    runtime=EventTime(110, EventTime.Unit.US),
                ),
            ]
        ),
    )
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
        profile=work_profile,
        deadline=190,
    )
    camera_task_2 = create_default_task(
        name="Camera_2",
        job=Job(name="Camera_2"),
        profile=work_profile,
        deadline=190,
    )
    camera_task_3 = create_default_task(
        name="Camera_3",
        job=Job(name="Camera_3"),
        profile=deepcopy(work_profile),
        deadline=300,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={camera_task_1: [], camera_task_2: [], camera_task_3: []},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())
    camera_task_3.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 15}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Run the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 3, "Incorrect length of placements retrieved."

    camera_task_1_placements = placements.get_placements(camera_task_1)
    assert len(camera_task_1_placements) == 1, "Placement for camera_task_1 not found."
    assert camera_task_1_placements[
        0
    ].is_placed(), "Placement for camera_task_1 failed."
    assert isinstance(
        camera_task_1_placements[0].execution_strategy, BatchStrategy
    ), "Incorrect strategy type for camera_task_1."
    assert (
        camera_task_1_placements[0].execution_strategy.batch_size == 2
    ), "Incorrect batch size for camera_task_1."

    camera_task_2_placements = placements.get_placements(camera_task_2)
    assert len(camera_task_2_placements) == 1, "Placement for camera_task_2 not found."
    assert camera_task_2_placements[
        0
    ].is_placed(), "Placement for camera_task_1 failed."
    assert isinstance(
        camera_task_2_placements[0].execution_strategy, BatchStrategy
    ), "Incorrect strategy type for camera_task_2."
    assert (
        camera_task_2_placements[0].execution_strategy.batch_size == 2
    ), "Incorrect batch size for camera_task_2."

    camera_task_3_placements = placements.get_placements(camera_task_3)
    assert len(camera_task_3_placements) == 1, "Placement for camera_task_3 not found."
    assert camera_task_3_placements[
        0
    ].is_placed(), "Placement for camera_task_3 failed."
    assert isinstance(
        camera_task_3_placements[0].execution_strategy, BatchStrategy
    ), "Incorrect strategy type for camera_task_3."
    assert (
        camera_task_3_placements[0].execution_strategy.batch_size == 1
    ), "Incorrect batch size for camera_task_3."


@pytest.mark.parametrize(
    "scheduler",
    [
        ILPScheduler(
            preemptive=False,
            runtime=EventTime.zero(),
            lookahead=EventTime.zero(),
            enforce_deadlines=True,
            batching=True,
            release_taskgraphs=True,
        ),
    ],
)
def test_ilp_batching_graph(scheduler):
    """Test that the ILP-based scheduler batches taskgraphs correctly."""
    # Create all the WorkProfiles for the tasks.
    camera_work_profile = WorkProfile(
        name="Camera_Profile",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 10}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 12}
                    ),
                    batch_size=2,
                    runtime=EventTime(110, EventTime.Unit.US),
                ),
            ]
        ),
    )
    perception_work_profile = WorkProfile(
        name="Perception_Profile",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 10}
                    ),
                    batch_size=1,
                    runtime=EventTime(200, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 12}
                    ),
                    batch_size=2,
                    runtime=EventTime(250, EventTime.Unit.US),
                ),
            ]
        ),
    )

    # Create the first TaskGraph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1"),
        profile=camera_work_profile,
        deadline=400,
        task_graph_name="TestTaskGraph_1",
    )
    perception_task_1 = create_default_task(
        name="Perception_1",
        job=Job(name="Perception_1"),
        profile=perception_work_profile,
        deadline=400,
        task_graph_name="TestTaskGraph_1",
    )
    task_graph_1 = TaskGraph(
        name="TestTaskGraph_1",
        tasks={camera_task_1: [perception_task_1], perception_task_1: []},
    )

    # Create the second TaskGraph.
    camera_task_2 = create_default_task(
        name="Camera_2",
        job=Job(name="Camera_2"),
        profile=camera_work_profile,
        deadline=400,
        task_graph_name="TestTaskGraph_2",
    )
    perception_task_2 = create_default_task(
        name="Perception_2",
        job=Job(name="Perception_2"),
        profile=perception_work_profile,
        deadline=400,
        task_graph_name="TestTaskGraph_2",
    )
    task_graph_2 = TaskGraph(
        name="TestTaskGraph_2",
        tasks={camera_task_2: [perception_task_2], perception_task_2: []},
    )

    # Create the Workload.
    workload = Workload.from_task_graphs(
        {"TestTaskGraph_1": task_graph_1, "TestTaskGraph_2": task_graph_2}
    )
    camera_task_1.release(EventTime.zero())
    camera_task_2.release(EventTime.zero())
    perception_task_1.release(EventTime.zero())
    perception_task_2.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources(resource_vector={Resource(name="CPU"): 15}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Run the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 4, "Incorrect number of placements."

    # Check that all the tasks have been placed.
    assert all(
        placement.is_placed() for placement in placements
    ), "All Tasks were expected to be placed."

    # Check that the dependency of the graphs is requested.
    camera_task_1_placement = placements.get_placements(camera_task_1)[0]
    perception_task_1_placement = placements.get_placements(perception_task_1)[0]
    assert (
        camera_task_1_placement.placement_time
        + camera_task_1_placement.execution_strategy.runtime
        <= perception_task_1_placement.placement_time
    ), "Incorrect placement time for perception_task_1."

    camera_task_2_placement = placements.get_placements(camera_task_2)[0]
    perception_task_2_placement = placements.get_placements(perception_task_2)[0]
    assert (
        camera_task_2_placement.placement_time
        + camera_task_2_placement.execution_strategy.runtime
        <= perception_task_2_placement.placement_time
    ), "Incorrect placement time for perception_task_2."

    # Check that all the tasks were placed in a batch size of 2.
    assert all(
        placement.execution_strategy.batch_size == 2 for placement in placements
    ), "All Tasks were expected to be placed in a batch size of 2."


@pytest.mark.parametrize(
    "scheduler",
    [
        pytest.param(
            TetriSchedGurobiScheduler(
                runtime=EventTime.zero(),
                batching=True,
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
        ),
        pytest.param(
            TetriSchedCPLEXScheduler(
                runtime=EventTime.zero(),
                batching=True,
                enforce_deadlines=True,
                time_discretization=EventTime(10, EventTime.Unit.US),
            ),
            marks=pytest.mark.skipif(
                os.getenv("GITHUB_ACTIONS") == "true",
                reason="CPLEX is not available in Github Actions.",
            ),
        ),
    ],
    ids=["TetriSchedGurobi", "TetriSchedCPLEX"],
)
def test_ilp_batching_time(scheduler):
    """Test that the ILP-based scheduler batches tasks and places them in order."""
    work_profile = WorkProfile(
        name="Camera_Profile",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 10}
                    ),
                    batch_size=16,
                    runtime=EventTime(110, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 10}
                    ),
                    batch_size=8,
                    runtime=EventTime(80, EventTime.Unit.US),
                ),
            ]
        ),
    )
    tasks = [
        create_default_task(
            name="Camera_{}".format(i),
            job=Job(name="Camera_{}".format(i)),
            profile=work_profile,
            deadline=190,
        )
        for i in range(24)
    ]
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={task: [] for task in tasks},
    )
    workload = Workload.from_task_graphs({"TestTaskGraph": task_graph})
    for task in tasks:
        task.release(EventTime.zero())

    # Create the Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 10}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Run the scheduler.
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)
    assert len(placements) == 24, "Incorrect length of placements retrieved."

    batch_16_placements = []
    batch_8_placements = []
    for placement in placements:
        if placement.execution_strategy.batch_size == 16:
            batch_16_placements.append(placement)
        elif placement.execution_strategy.batch_size == 8:
            batch_8_placements.append(placement)
        else:
            assert False, (
                f"Incorrect batch size "
                f"({placement.execution_strategy.batch_size}) retrieved."
            )

    assert (
        batch_16_placements[0].placement_time < batch_8_placements[0].placement_time
    ), "Incorrect placement order."
