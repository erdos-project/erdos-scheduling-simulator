from logging import raiseExceptions

import pytest

from schedulers import BranchPredictionScheduler, EDFScheduler, LSFScheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    BranchPredictionPolicy,
    Job,
    Resource,
    Resources,
    TaskGraph,
    Workload,
)


def test_edf_scheduler_success():
    """Scenario:

    EDF scheduler successfully schedules the tasks across a set of WorkerPools
    according to the resource requirements.
    """
    edf_scheduler = EDFScheduler()

    # Create the tasks and the TaskGraph.
    task_cpu = create_default_task(
        name="cpu_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        deadline=200,
    )
    task_cpu.release(EventTime(1, EventTime.Unit.US))
    task_gpu = create_default_task(
        name="gpu_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 1}
        ),
        deadline=50,
    )
    task_gpu.release(EventTime(1, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_cpu: [], task_gpu: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker_one = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_pool_one = WorkerPool(name="WorkerPool_1", workers=[worker_one])
    worker_two = Worker(
        name="Worker",
        resources=Resources({Resource(name="GPU"): 1}),
    )
    worker_pool_two = WorkerPool(name="WorkerPool_2", workers=[worker_two])

    # Schedule the tasks.
    placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool_one, worker_pool_two]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."

    task_gpu_placement = placements.get_placement(task_gpu)
    assert task_gpu_placement is not None, "The task was not found in placements."
    assert (
        task_gpu_placement.worker_pool_id == worker_pool_two.id
    ), "Incorrect placement of the task on the WorkerPool."

    task_cpu_placement = placements.get_placement(task_cpu)
    assert task_cpu_placement is not None, "The task was not found in placements."
    assert (
        task_cpu_placement.worker_pool_id == worker_pool_one.id
    ), "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_limited_resources():
    """Scenario:

    EDF scheduler chooses the higher-priority task if only enough resources
    for one task.
    """
    edf_scheduler = EDFScheduler()

    # Create the tasks and the TaskGraph.
    task_lower_priority = create_default_task(name="task_low_priority", deadline=200)
    task_lower_priority.release(EventTime(1, EventTime.Unit.US))
    task_higher_priority = create_default_task(name="task_high_priority", deadline=50)
    task_higher_priority.release(EventTime(1, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_lower_priority: [], task_higher_priority: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."

    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        task_higher_priority_placement.worker_pool_id == worker_pool.id
    ), "Incorrect placement of the task on the WorkerPool."

    task_lower_priority_placement = placements.get_placement(task_lower_priority)
    assert (
        task_lower_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        not task_lower_priority_placement.is_placed()
    ), "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_non_preemptive_higher_priority():
    """Scenario:

    Non-Preemptive EDF scheduler does not preempt already placed task even if
    a new task with a closer deadline is added.
    """
    edf_scheduler = EDFScheduler(preemptive=False)

    # Create the tasks and the TaskGraph.
    task_lower_priority = create_default_task(name="task_low_priority", deadline=200)
    task_lower_priority.release(EventTime(1, EventTime.Unit.US))
    task_higher_priority = create_default_task(name="task_high_priority", deadline=50)
    task_graph = TaskGraph(tasks={task_lower_priority: [], task_higher_priority: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the lower priority task.
    placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )

    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is None
    ), "The task was not found in placements."
    task_lower_priority_placement = placements.get_placement(task_lower_priority)
    assert (
        task_lower_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        task_lower_priority_placement.worker_pool_id == worker_pool.id
    ), "Incorrect placement of the low priority task."
    worker_pool.place_task(task_lower_priority)

    task_higher_priority.release(EventTime(2, EventTime.Unit.US))
    # Schedule the higher priority task.
    placements = edf_scheduler.schedule(
        EventTime(2, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        not task_higher_priority_placement.is_placed()
    ), "Incorrect placement of the high priority task."


def test_edf_scheduler_preemptive_higher_priority():
    """Scenario:

    Preemptive EDF scheduler preempts already placed task when a new task with
    a closer deadline is added.
    """
    edf_scheduler = EDFScheduler(preemptive=True)

    # Create the tasks and the TaskGraph.
    task_lower_priority = create_default_task(name="task_low_priority", deadline=200)
    task_lower_priority.release(EventTime(1, EventTime.Unit.US))
    task_higher_priority = create_default_task(name="task_high_priority", deadline=50)
    task_graph = TaskGraph(tasks={task_lower_priority: [], task_higher_priority: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the lower priority task.
    placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )

    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is None
    ), "The task was not found in placements."
    task_lower_priority_placement = placements.get_placement(task_lower_priority)
    assert (
        task_lower_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        task_lower_priority_placement.worker_pool_id == worker_pool.id
    ), "Incorrect placement of the low priority task."

    worker_pool.place_task(task_lower_priority)
    task_higher_priority.release(EventTime(2, EventTime.Unit.US))

    # Schedule the higher priority task.
    placements = edf_scheduler.schedule(
        EventTime(2, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )

    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        task_higher_priority_placement.worker_pool_id == worker_pool.id
    ), "Incorrect placement of the high priority task."
    task_lower_priority_placement = placements.get_placement(task_lower_priority)
    assert (
        task_lower_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        not task_lower_priority_placement.is_placed()
    ), "Incorrect placement of the low priority task."


def test_lsf_scheduler_success():
    """Scenario:

    LSF scheduler successfully schedules the tasks across a set of WorkerPools
    according to the resource requirements.
    """
    lsf_scheduler = LSFScheduler()

    # Create the tasks and the TaskGraph.
    task_cpu = create_default_task(
        name="cpu_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        deadline=200,
    )
    task_cpu.release(EventTime(1, EventTime.Unit.US))
    task_gpu = create_default_task(
        name="gpu_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 1}
        ),
        deadline=50,
    )
    task_gpu.release(EventTime(1, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_cpu: [], task_gpu: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker_one = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_pool_one = WorkerPool(name="WorkerPool_1", workers=[worker_one])
    worker_two = Worker(
        name="Worker",
        resources=Resources({Resource(name="GPU"): 1}),
    )
    worker_pool_two = WorkerPool(name="WorkerPool_2", workers=[worker_two])

    # Schedule the tasks.
    placements = lsf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool_one, worker_pool_two]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."

    task_gpu_placement = placements.get_placement(task_gpu)
    assert task_gpu_placement is not None, "The task was not found in placements."
    assert (
        task_gpu_placement.worker_pool_id == worker_pool_two.id
    ), "Incorrect placement of the task on the WorkerPool."

    task_cpu_placement = placements.get_placement(task_cpu)
    assert task_cpu_placement is not None, "The task was not found in placements."
    assert (
        task_cpu_placement.worker_pool_id == worker_pool_one.id
    ), "Incorrect placement of the task on the WorkerPool."


def test_lsf_scheduler_limited_resources():
    """Scenario:

    EDF scheduler chooses the higher-priority task if only enough resources
    for one task.
    """
    lsf_scheduler = LSFScheduler()

    # Create the tasks and the TaskGraph.
    task_lower_priority = create_default_task(name="task_low_priority", deadline=200)
    task_lower_priority.release(EventTime(50, EventTime.Unit.US))
    task_lower_priority.update_remaining_time(EventTime(100, EventTime.Unit.US))
    task_higher_priority = create_default_task(name="task_high_priority", deadline=220)
    task_higher_priority.release(EventTime(50, EventTime.Unit.US))
    task_higher_priority.update_remaining_time(EventTime(150, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_lower_priority: [], task_higher_priority: []})
    workload = Workload.from_task_graphs({"test_task_graph": task_graph})

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    placements = lsf_scheduler.schedule(
        EventTime(50, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."

    task_higher_priority_placement = placements.get_placement(task_higher_priority)
    assert (
        task_higher_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        task_higher_priority_placement.worker_pool_id == worker_pool.id
    ), "Incorrect placement of the task on the WorkerPool."

    task_lower_priority_placement = placements.get_placement(task_lower_priority)
    assert (
        task_lower_priority_placement is not None
    ), "The task was not found in placements."
    assert (
        not task_lower_priority_placement.is_placed()
    ), "Incorrect placement of the task on the WorkerPool."


def test_branch_prediction_scheduler_slack():
    """Test that the BranchPredictionScheduler calculates the slack
    correctly in case of conditional execution of nodes."""
    # Create the tasks and the TaskGraph.
    perception_task = create_default_task(
        name="Perception@0",
        job=Job(
            name="Perception",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        ),
        runtime=1000,
    )
    prediction_task_0 = create_default_task(
        name="Prediction@0",
        job=Job(
            name="Prediction_0",
            runtime=EventTime(2000, EventTime.Unit.US),
            probability=1.0,
        ),
        runtime=2000,
    )
    prediction_task_1 = create_default_task(
        name="Prediction@1",
        job=Job(name="Prediction_1", runtime=EventTime(4000, EventTime.Unit.US)),
        runtime=4000,
    )
    planning_task_0 = create_default_task(
        name="Planning@0",
        job=Job(
            name="Planning_0",
            runtime=EventTime(6000, EventTime.Unit.US),
            probability=0.0,
        ),
        runtime=6000,
    )
    planning_task_1 = create_default_task(
        name="Planning@1",
        job=Job(name="Planning_1", runtime=EventTime(8000, EventTime.Unit.US)),
        runtime=8000,
    )
    perception_terminal_task = create_default_task(
        name="Perception_End@0",
        job=Job(name="Perception_End", runtime=EventTime.zero()),
        runtime=0,
        deadline=15000,
    )
    task_graph = TaskGraph(
        {
            perception_task: [prediction_task_0, planning_task_0],
            prediction_task_0: [prediction_task_1],
            planning_task_0: [planning_task_1],
            prediction_task_1: [perception_terminal_task],
            planning_task_1: [perception_terminal_task],
            perception_terminal_task: [],
        }
    )

    # Assert schedulers with different policies and remaining time.
    worst_case_scheduler = BranchPredictionScheduler(
        policy=BranchPredictionPolicy.WORST_CASE
    )
    slack = worst_case_scheduler.compute_slack(
        EventTime.zero(), task_graph, "test_task_graph"
    )
    assert slack == EventTime.zero(), "Incorrect slack returned."

    best_case_scheduler = BranchPredictionScheduler(
        policy=BranchPredictionPolicy.BEST_CASE
    )
    slack = best_case_scheduler.compute_slack(
        EventTime.zero(), task_graph, "test_task_graph"
    )
    assert slack == EventTime(8, EventTime.Unit.MS), "Incorrect slack returned."

    random_scheduler = BranchPredictionScheduler(policy=BranchPredictionPolicy.RANDOM)
    slack = random_scheduler.compute_slack(
        EventTime.zero(), task_graph, "test_task_graph"
    )
    assert (
        slack == EventTime(8, EventTime.Unit.MS) or slack == EventTime.zero()
    ), "Incorrect slack returned."
