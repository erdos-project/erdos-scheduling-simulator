from workload import TaskGraph, Resource, Resources
from schedulers import EDFScheduler, LSFScheduler
from schedulers.ilp_scheduler import ILPBaseScheduler
from schedulers.z3_scheduler import Z3Scheduler
from workers import Worker, WorkerPool

from tests.test_tasks import __create_default_task


def test_ilp_scheduler_success():
    """Scenario:

    EDF scheduler successfully schedules the tasks across a set of WorkerPools
    according to the resource requirements.
    """
    ilp_scheduler = ILPBaseScheduler(Z3Scheduler)

    # Create the tasks and the TaskGraph.
    task_cpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}),
                                     deadline=200.0)
    task_gpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 1}),
                                     deadline=50.0)
    task_graph = TaskGraph()

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
    _, placements = ilp_scheduler.schedule(
        1.0,
        released_tasks=[task_cpu, task_gpu],
        task_graph=task_graph,
        worker_pools=[worker_pool_one, worker_pool_two],
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[1][0] == task_gpu,\
        "Incorrect task received in the placement."
    assert placements[1][1] == worker_pool_two.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[0][0] == task_cpu,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool_one.id,\
        "Incorrect placement of the task on the WorkerPool."


def test_ilp_scheduler_limited_resources():
    """Scenario:

    EDF scheduler chooses the higher-priority task if only enough resources
    for one task.
    """
    ilp_scheduler = ILPBaseScheduler(Z3Scheduler)

    # Create the tasks and the TaskGraph.
    task_lower_priority = __create_default_task(deadline=200.0)
    task_lower_priority.update_remaining_time(100.0)
    task_higher_priority = __create_default_task(deadline=220.0)
    task_higher_priority.update_remaining_time(150.0)
    task_graph = TaskGraph()

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1
        }),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    _, placements = ilp_scheduler.schedule(
        50.0,
        released_tasks=[task_lower_priority, task_higher_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])

    # TODO (Justin): No behavior for loadshedding
    assert placements is None, "Doesn't detect workload is unschedulable."

    # assert len(placements) == 2, "Incorrect length of task placements."
    # assert placements[1][0] == task_higher_priority,\
    #     "Incorrect task received in the placement."
    # assert placements[1][1] == worker_pool.id,\
    #     "Incorrect placement of the task on the WorkerPool."
    # assert placements[0][0] == task_lower_priority,\
    #     "Incorrect task received in the placement."
    # assert placements[0][1] is None,\
    #     "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_success():
    """Scenario:

    EDF scheduler successfully schedules the tasks across a set of WorkerPools
    according to the resource requirements.
    """
    edf_scheduler = EDFScheduler()

    # Create the tasks and the TaskGraph.
    task_cpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}),
                                     deadline=200.0)
    task_gpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 1}),
                                     deadline=50.0)
    task_graph = TaskGraph()

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
        1.0,
        released_tasks=[task_cpu, task_gpu],
        task_graph=task_graph,
        worker_pools=[worker_pool_one, worker_pool_two],
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_gpu,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool_two.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_cpu,\
        "Incorrect task received in the placement."
    assert placements[1][1] == worker_pool_one.id,\
        "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_limited_resources():
    """Scenario:

    EDF scheduler chooses the higher-priority task if only enough resources
    for one task.
    """
    edf_scheduler = EDFScheduler()

    # Create the tasks and the TaskGraph.
    task_lower_priority = __create_default_task(deadline=200.0)
    task_higher_priority = __create_default_task(deadline=50.0)
    task_graph = TaskGraph()

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1
        }),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    _, placements = edf_scheduler.schedule(
        1.0,
        released_tasks=[task_lower_priority, task_higher_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool],
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_higher_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_lower_priority,\
        "Incorrect task received in the placement."
    assert placements[1][1] is None,\
        "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_non_preemptive_higher_priority():
    """Scenario:

    Non-Preemptive EDF scheduler does not preempt already placed task even if
    a new task with a closer deadline is added.
    """
    edf_scheduler = EDFScheduler(preemptive=False)

    # Create the tasks and the TaskGraph.
    task_lower_priority = __create_default_task(deadline=200.0)
    task_higher_priority = __create_default_task(deadline=50.0)
    task_graph = TaskGraph()

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1
        }),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the lower priority task.
    _, placements = edf_scheduler.schedule(
        1.0,
        released_tasks=[task_lower_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])
    assert len(placements) == 1, "Incorrect length of task placements."
    assert placements[0][0] == task_lower_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool.id,\
        "Incorrect placement of the task on the WorkerPool."
    worker_pool.place_task(task_lower_priority)

    # Schedule the higher priority task.
    _, placements = edf_scheduler.schedule(
        1.0,
        released_tasks=[task_higher_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])
    assert len(placements) == 1, "Incorrect length of task placements."
    assert placements[0][0] == task_higher_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] is None,\
        "Incorrect placement of the task on the WorkerPool."


def test_edf_scheduler_preemptive_higher_priority():
    """Scenario:

    Preemptive EDF scheduler preempts already placed task when a new task with
    a closer deadline is added.
    """
    edf_scheduler = EDFScheduler(preemptive=True)

    # Create the tasks and the TaskGraph.
    task_lower_priority = __create_default_task(deadline=200.0)
    task_higher_priority = __create_default_task(deadline=50.0)
    task_graph = TaskGraph()

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1
        }),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the lower priority task.
    _, placements = edf_scheduler.schedule(
        1.0,
        released_tasks=[task_lower_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])
    assert len(placements) == 1, "Incorrect length of task placements."
    assert placements[0][0] == task_lower_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool.id,\
        "Incorrect placement of the task on the WorkerPool."
    worker_pool.place_task(task_lower_priority)

    # Schedule the higher priority task.
    _, placements = edf_scheduler.schedule(
        1.0,
        released_tasks=[task_higher_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_higher_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_lower_priority,\
        "Incorrect task received in the placement."
    assert placements[1][1] is None,\
        "Incorrect placement of the task on the WorkerPool."


def test_lsf_scheduler_success():
    """Scenario:

    LSF scheduler successfully schedules the tasks across a set of WorkerPools
    according to the resource requirements.
    """
    lsf_scheduler = LSFScheduler()

    # Create the tasks and the TaskGraph.
    task_cpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}),
                                     deadline=200.0)
    task_gpu = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 1}),
                                     deadline=50.0)
    task_graph = TaskGraph()

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
        1.0,
        released_tasks=[task_cpu, task_gpu],
        task_graph=task_graph,
        worker_pools=[worker_pool_one, worker_pool_two],
    )
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_gpu,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool_two.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_cpu,\
        "Incorrect task received in the placement."
    assert placements[1][1] == worker_pool_one.id,\
        "Incorrect placement of the task on the WorkerPool."


def test_lsf_scheduler_limited_resources():
    """Scenario:

    EDF scheduler chooses the higher-priority task if only enough resources
    for one task.
    """
    lsf_scheduler = LSFScheduler()

    # Create the tasks and the TaskGraph.
    task_lower_priority = __create_default_task(deadline=200.0)
    task_lower_priority.update_remaining_time(100.0)
    task_higher_priority = __create_default_task(deadline=220.0)
    task_higher_priority.update_remaining_time(150.0)
    task_graph = TaskGraph()

    # Create the WorkerPool.
    worker = Worker(
        name="Worker",
        resources=Resources({
            Resource(name="CPU"): 1,
            Resource(name="GPU"): 1
        }),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker])

    # Schedule the tasks.
    _, placements = lsf_scheduler.schedule(
        50.0,
        released_tasks=[task_lower_priority, task_higher_priority],
        task_graph=task_graph,
        worker_pools=[worker_pool])
    assert len(placements) == 2, "Incorrect length of task placements."
    assert placements[0][0] == task_higher_priority,\
        "Incorrect task received in the placement."
    assert placements[0][1] == worker_pool.id,\
        "Incorrect placement of the task on the WorkerPool."
    assert placements[1][0] == task_lower_priority,\
        "Incorrect task received in the placement."
    assert placements[1][1] is None,\
        "Incorrect placement of the task on the WorkerPool."
