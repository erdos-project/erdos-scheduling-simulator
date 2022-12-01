import pytest

from schedulers import ILPScheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Job, Resource, Resources, TaskGraph, Workload


def test_ilp_scheduling_success_basic():
    """Test that a single task can be successfully scheduled with enough resources."""
    # Create the tasks and the graph.
    camera_task_1 = create_default_task(
        name="Camera_1",
        job=Job(name="Camera_1", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
        timestamp=0,
        deadline=1100,
    )
    task_graph = TaskGraph(tasks={camera_task_1: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})
    camera_task_1.release(EventTime.zero())

    # Create the workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 2}),
    )
    worker_pool_1 = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool_1])

    # Create the scheduler.
    scheduler = ILPScheduler(
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
    assert (
        camera_task_placement.placement_time + camera_task_1.remaining_time
        <= camera_task_1.deadline
    ), "Incorrect start time retrieved."


def test_ilp_scheduling_multiple_tasks_different_resources():
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
    task_graph = TaskGraph(tasks={cpu_task: [], cpu_gpu_task: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})
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
    scheduler = ILPScheduler(
        preemptive=False,
        runtime=EventTime.zero(),
        lookahead=EventTime.zero(),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(EventTime.zero(), workload, worker_pools)

    assert len(placements) == 2, "Incorrect length of placements retrieved."
    
    cpu_task_placement = placements.get_placement(cpu_task)
    assert cpu_task_placement is not None, "The task was expected to be placed."
    assert (
        cpu_task_placement.worker_pool_id == worker_pool_1.id
    ), "Incorrect WorkerPoolID retrieved."
    assert (
        cpu_task_placement.placement_time + cpu_task.remaining_time
        <= cpu_task.deadline
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
