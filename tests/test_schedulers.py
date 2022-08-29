from logging import raiseExceptions

from schedulers import EDFScheduler, GurobiScheduler, LSFScheduler, Z3Scheduler
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources, TaskGraph
from workload.workload import Workload


def test_gurobi_scheduler_success():
    """Scenario:

    ILP Gurobi scheduler successfully schedules the tasks across a set of
    WorkerPools according to the resource requirements.
    """
    scheduler = GurobiScheduler()

    # Create the tasks and the TaskGraph.
    task_cpu = create_default_task(
        name="cpu_task",
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        deadline=200,
    )
    task_cpu.release(EventTime(1, EventTime.Unit.US))
    task_cpu_gpu = create_default_task(
        name="gpu_task",
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        ),
        deadline=50,
    )
    task_cpu_gpu.release(EventTime(1, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_cpu: [], task_cpu_gpu: []})

    # Create the WorkerPool.
    worker_one = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1}),
    )
    worker_pool_one = WorkerPool(name="WorkerPool_1", workers=[worker_one])

    worker_two = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool_two = WorkerPool(name="WorkerPool_2", workers=[worker_two])

    # Schedule the tasks.
    _, placements = scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        task_graph=task_graph,
        worker_pools=WorkerPools([worker_pool_one, worker_pool_two]),
    )

    assert len(placements) == 2, "Incorrect length of task placements."

    if placements[1][0] == task_cpu_gpu and placements[0][0] == task_cpu:
        assert (
            placements[1][1] == worker_pool_two.id
        ), "Incorrect placement of the CPU & GPU task on the WorkerPool."
        assert (
            placements[0][1] == worker_pool_one.id
        ), "Incorrect placement of the CPU task on the WorkerPool."
    elif placements[0][0] == task_cpu_gpu and placements[1][0] == task_cpu:
        assert (
            placements[0][1] == worker_pool_two.id
        ), "Incorrect placement of the CPU & GPU task on the WorkerPool."
        assert (
            placements[1][1] == worker_pool_one.id
        ), "Incorrect placement of the CPU task on the WorkerPool."
    else:
        raiseExceptions(
            "Incorrect placements: two tasks arent the same as the ones placed"
        )


def test_gurobi_scheduler_limited_resources():
    """Scenario:

    ILP Gurobi scheduler recognizes the workload is not schedulable.
    """
    scheduler = GurobiScheduler()

    # Create the tasks and the TaskGraph.
    task_lower_priority = create_default_task(name="task_low_priority", deadline=200)
    task_lower_priority.release(EventTime(50, EventTime.Unit.US))
    task_lower_priority.update_remaining_time(EventTime(100, EventTime.Unit.US))
    task_higher_priority = create_default_task(name="task_high_priority", deadline=220)
    task_higher_priority.release(EventTime(50, EventTime.Unit.US))
    task_higher_priority.update_remaining_time(EventTime(150, EventTime.Unit.US))
    task_graph = TaskGraph(tasks={task_lower_priority: [], task_higher_priority: []})

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    _, placements = scheduler.schedule(
        EventTime(50, EventTime.Unit.US),
        task_graph=task_graph,
        worker_pools=WorkerPools([worker_pool]),
    )

    assert len(placements) == 2, "Didn't try to schedule both tasks."
    if placements[0][0] == task_higher_priority:
        assert placements[0][1] is not None, "Didn't place the high priority task."
        assert placements[1][1] is None, "Placed the low priority task."
    else:
        assert placements[1][1] is not None, "Didn't place the high priority task."
        assert placements[0][1] is None, "Placed the low priority task."


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
    _, placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool_one, worker_pool_two]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_gpu, "Incorrect task received in the placement."
    assert (
        placements[0][1] == worker_pool_two.id
    ), "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_cpu, "Incorrect task received in the placement."
    assert (
        placements[1][1] == worker_pool_one.id
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
    _, placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert (
        placements[0][0] == task_higher_priority
    ), "Incorrect task received in the placement."
    assert (
        placements[0][1] == worker_pool.id
    ), "Incorrect placement of the task on the WorkerPool."
    assert (
        placements[1][0] == task_lower_priority
    ), "Incorrect task received in the placement."
    assert (
        placements[1][1] is None
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
    _, placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    for (task, placement, _) in placements:
        if task == task_higher_priority:
            assert placement is None, "Incorrect placement of the high priority task."
        elif task == task_lower_priority:
            assert (
                placement == worker_pool.id
            ), "Incorrect placement of the low priority task."
    worker_pool.place_task(task_lower_priority)

    task_higher_priority.release(EventTime(2, EventTime.Unit.US))
    # Schedule the higher priority task.
    _, placements = edf_scheduler.schedule(
        EventTime(2, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    for (task, placement, _) in placements:
        if task == task_higher_priority:
            assert placement is None, "Incorrect placement of the high priority task."


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
    _, placements = edf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    for (task, placement, _) in placements:
        if task == task_higher_priority:
            assert placement is None, "Incorrect placement of the high priority task."
        elif task == task_lower_priority:
            assert (
                placement == worker_pool.id
            ), "Incorrect placement of the low priority task."
    worker_pool.place_task(task_lower_priority)
    task_higher_priority.release(EventTime(2, EventTime.Unit.US))
    # # Schedule the higher priority task.
    _, placements = edf_scheduler.schedule(
        EventTime(2, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    for (task, placement, _) in placements:
        if task == task_lower_priority:
            assert placement is None, "Incorrect placement of the low priority task."
        elif task == task_higher_priority:
            assert (
                placement == worker_pool.id
            ), "Incorrect placement of the high priority task."


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
    _, placements = lsf_scheduler.schedule(
        EventTime(1, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool_one, worker_pool_two]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_gpu, "Incorrect task received in the placement."
    assert (
        placements[0][1] == worker_pool_two.id
    ), "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_cpu, "Incorrect task received in the placement."
    assert (
        placements[1][1] == worker_pool_one.id
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
    _, placements = lsf_scheduler.schedule(
        EventTime(50, EventTime.Unit.US),
        workload=workload,
        worker_pools=WorkerPools([worker_pool]),
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert (
        placements[0][0] == task_higher_priority
    ), "Incorrect task received in the placement."
    assert (
        placements[0][1] == worker_pool.id
    ), "Incorrect placement of the task on the WorkerPool."
    assert (
        placements[1][0] == task_lower_priority
    ), "Incorrect task received in the placement."
    assert (
        placements[1][1] is None
    ), "Incorrect placement of the task on the WorkerPool."
