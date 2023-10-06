import pytest

# We will not be building the TetriSchedScheduler in Github Actions
# so we skip this module in the CI.
tetrisched = pytest.importorskip("tetrisched_py")

try:
    from schedulers import TetriSchedScheduler
except ImportError:
    pass
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources, TaskGraph, Workload


def test_tetrisched_partition_creation_non_slot_failure():
    """Tests that the TetrischedScheduler raises a ValueError when a WorkerPool
    contains a Worker with a non-Slot resource."""
    # Construct a WorkerPool with a Worker that has a non-Slot resource.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="GPU"): 1})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Create the TetriSchedScheduler and check that it raises a ValueError.
    scheduler = TetriSchedScheduler()
    with pytest.raises(ValueError):
        scheduler.construct_partitions(worker_pools=worker_pools)


def test_tetrisched_partition_creation_success():
    """Tests that the Partition objects are created correctly."""
    # Create the WorkerPools with two workers and two slots each.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_2 = Worker(
        name="worker_2", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_pool_1 = WorkerPool(name="worker_pool_1", workers=[worker_1, worker_2])

    worker_3 = Worker(
        name="worker_3", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_4 = Worker(
        name="worker_4", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_pool_2 = WorkerPool(name="worker_pool_2", workers=[worker_3, worker_4])

    worker_pools = WorkerPools(worker_pools=[worker_pool_1, worker_pool_2])

    # Create the TetriSchedScheduler and check that it returns a single Partitions
    # object with four partitions that each have 2 slots.
    scheduler = TetriSchedScheduler()
    partitions = scheduler.construct_partitions(worker_pools=worker_pools)

    assert len(partitions) == 4, "The number of Partition objects is not 2."
    for partition in partitions.getPartitions():
        assert len(partition) == 2, "The number of slots in a Partition is not 2."


def test_tetrisched_task_strl_no_slot_fail():
    """Tests that a Task without Slot requirements fails."""
    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the Task.
    task = create_default_task(
        name="task",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="GPU"): 1}),
    )

    # Construct the scheduler and the partitions.
    scheduler = TetriSchedScheduler()
    partitions = scheduler.construct_partitions(worker_pools=worker_pools)
    with pytest.raises(ValueError):
        scheduler.construct_task_strl(
            current_time=EventTime.zero(), task=task, partitions=partitions
        )


def test_tetrisched_task_choice_strl_generation():
    """Tests that a Task's STRL expression is correct."""
    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the Task.
    task = create_default_task(
        name="task",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="Slot"): 1}),
    )

    # Construct the scheduler and the partitions.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(10, EventTime.Unit.US)
    )
    partitions = scheduler.construct_partitions(worker_pools=worker_pools)
    task_strl = scheduler.construct_task_strl(
        current_time=EventTime.zero(), task=task, partitions=partitions
    )

    # Ensure that the type and number of choices are correct.
    assert (
        task_strl.getType() == tetrisched.strl.EXPR_MAX
    ), f"Incorrect type of the root of STRL subtree: {task_strl.getType()}"
    assert task_strl.getNumChildren() == 3, "Incorrect number of children in STRL."
    for child_expr in task_strl.getChildren():
        assert (
            child_expr.getType() == tetrisched.strl.EXPR_CHOOSE
        ), f"Incorrect type of child of STRL subtree: {child_expr.getType()}"


def test_tetrisched_task_graph_strl_generation_simple():
    """Tests that a linear chain TaskGraph's STRL expression is correct."""
    # Create a chain TaskGraph with two tasks.
    task_one = create_default_task(
        name="task_one",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="Slot"): 1}),
    )
    task_two = create_default_task(
        name="task_two",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="Slot"): 1}),
    )
    task_graph = TaskGraph(
        name=task_one.task_graph, tasks={task_one: [task_two], task_two: []}
    )

    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 2})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the scheduler and the partitions.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(10, EventTime.Unit.US)
    )
    partitions = scheduler.construct_partitions(worker_pools=worker_pools)

    # Construct the STRL expression for the TaskGraph.
    task_strls = {}
    task_graph_strl = scheduler._construct_task_graph_strl(
        current_time=EventTime.zero(),
        task=task_one,
        task_graph=task_graph,
        partitions=partitions,
        task_strls=task_strls,
    )

    # Ensure that STRL expressions were constructed for all tasks.
    assert len(task_strls) == 2, "Incorrect number of STRL expressions."

    # Ensure that the generated STRL expression is correct.
    assert (
        task_graph_strl.getType() == tetrisched.strl.EXPR_LESSTHAN
    ), f"Incorrect type of the root of STRL subtree: {task_graph_strl.getType()}"
    assert (
        task_graph_strl.getNumChildren() == 2
    ), "Incorrect number of children in STRL."
    task_one_strl, task_two_strl = task_graph_strl.getChildren()
    assert (
        task_one_strl.getType() == tetrisched.strl.EXPR_MAX
    ), f"Incorrect type: {task_one_strl.getType()}"
    assert task_one_strl.getNumChildren() == 3, "Incorrect number of children in STRL."
    assert (
        task_two_strl.getType() == tetrisched.strl.EXPR_MAX
    ), f"Incorrect type: {task_two_strl.getType()}"
    assert task_two_strl.getNumChildren() == 3, "Incorrect number of children in STRL."


def test_two_tasks_correctly_scheduled():
    """Tests that two tasks without dependencies are correctly
    scheduled on a single Worker."""
    # Create two tasks.
    task_1 = create_default_task(
        name="task_1",
        runtime=20,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="Slot"): 1}),
    )
    task_2 = create_default_task(
        name="task_2",
        runtime=20,
        deadline=30,
        resource_requirements=Resources(resource_vector={Resource(name="Slot"): 1}),
    )
    task_graph = TaskGraph(name=task_1.task_graph, tasks={task_1: [], task_2: []})
    workload = Workload.from_task_graphs(task_graphs={task_graph.name: task_graph})

    # Release the tasks.
    task_1.release(EventTime.zero())
    task_2.release(EventTime.zero())

    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 1})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the scheduler and invoke it at the current time.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(10, EventTime.Unit.US)
    )
    scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
