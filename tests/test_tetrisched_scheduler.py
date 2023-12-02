import pytest

# We will not be building the TetriSchedScheduler in Github Actions
# so we skip this module in the CI.
tetrisched = pytest.importorskip("tetrisched_py")

try:
    from schedulers.tetrisched_scheduler import Partitions, TetriSchedScheduler
except ImportError:
    pass
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Resource,
    Resources,
    TaskGraph,
    Workload,
    WorkProfile,
)


def test_tetrisched_partition_creation_non_slot_failure():
    """Tests that the TetrischedScheduler raises a ValueError when a WorkerPool
    contains a Worker with a non-Slot resource."""
    # Construct a WorkerPool with a Worker that has a non-Slot resource.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="GPU"): 1})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Create the Partitions and check that it raises a ValueError.
    with pytest.raises(NotImplementedError):
        Partitions(worker_pools=worker_pools)


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

    # Create the Partitions and check that it returns a single Partitions
    # object with four partitions that each have 2 slots.
    partitions = Partitions(worker_pools=worker_pools)

    assert len(partitions) == 1, "The number of Partition objects is 1."
    slot_resource = Resource(name="Slot", _id="any")
    for partition in partitions[slot_resource].getPartitions():
        assert len(partition) == 2, "The quantity of the Partition is not 2."


def test_tetrisched_partition_creation_with_different_resource_types():
    """Tests that the Partition objects with different resource types are created
    correctly."""
    # Create the WorkerPools with two workers and two slots each.
    worker_1 = Worker(
        name="worker_1",
        resources=Resources(resource_vector={Resource(name="Slot_1"): 2}),
    )
    worker_2 = Worker(
        name="worker_2",
        resources=Resources(resource_vector={Resource(name="Slot_2"): 2}),
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1, worker_2])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Create the Partitions and check that it returns two Partitions object with
    # one partition each.
    partitions = Partitions(worker_pools=worker_pools)

    assert len(partitions) == 2, "The number of Partition objects is not 2."
    slot_1_resource = Resource(name="Slot_1", _id="any")
    assert partitions[slot_1_resource] is not None, "Slot_1 not found in Partitions."
    assert (
        len(partitions[slot_1_resource].getPartitions()) == 1
    ), "The number of Partitions is not 1."
    assert (
        len(partitions[slot_1_resource].getPartitions()[0]) == 2
    ), "The quantity of the Partition is not 2."

    slot_2_resource = Resource(name="Slot_2", _id="any")
    assert partitions[slot_2_resource] is not None, "Slot_2 not found in Partitions."
    assert (
        len(partitions[slot_2_resource].getPartitions()) == 1
    ), "The number of Partitions is not 1."
    assert (
        len(partitions[slot_2_resource].getPartitions()[0]) == 2
    ), "The quantity of the Partition is not 2."


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
    scheduler = TetriSchedScheduler(enforce_deadlines=True)
    partitions = Partitions(worker_pools=worker_pools)
    with pytest.raises(NotImplementedError):
        scheduler.construct_task_strl(
            current_time=EventTime.zero(),
            task=task,
            partitions=partitions,
            placement_times_and_rewards=[
                (EventTime(x, EventTime.Unit.US), 1) for x in range(0, 30, 10)
            ],
        )


def test_tetrisched_task_choice_strl_generation_with_single_strategy():
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
        time_discretization=EventTime(10, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    partitions = Partitions(worker_pools=worker_pools)
    task_strl = scheduler.construct_task_strl(
        current_time=EventTime.zero(),
        task=task,
        partitions=partitions,
        placement_times_and_rewards=[
            (EventTime.zero(), 1),
            (EventTime(10, EventTime.Unit.US), 1),
            (EventTime(20, EventTime.Unit.US), 1),
            (EventTime(30, EventTime.Unit.US), 1),
        ],
    )

    # Ensure that the type and number of choices are correct.
    assert (
        task_strl.getType() == tetrisched.strl.EXPR_WINDOWED_CHOOSE
    ), f"Incorrect type of the root of STRL subtree: {task_strl.getType()}"
    assert task_strl.getNumChildren() == 0, "Incorrect number of children in STRL."


def test_tetrisched_task_choice_strl_generation_with_multiple_strategies():
    """Tests that the STRL expression is correct when there are multiple strategies."""
    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1",
        resources=Resources(resource_vector={Resource(name="Slot_1"): 2}),
    )
    worker_2 = Worker(
        name="worker_2",
        resources=Resources(resource_vector={Resource(name="Slot_2"): 2}),
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1, worker_2])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the Task.
    task = create_default_task(
        name="task",
        profile=WorkProfile(
            name="task_work_profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_1"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(10, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_2"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(5, EventTime.Unit.US),
                    ),
                ],
            ),
        ),
        deadline=30,
    )

    # Construct the scheduler and the partitions.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(5, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    partitions = Partitions(worker_pools=worker_pools)
    task_strl = scheduler.construct_task_strl(
        current_time=EventTime.zero(),
        task=task,
        partitions=partitions,
        placement_times_and_rewards=[
            (EventTime(x, EventTime.Unit.US), 1) for x in range(0, 30, 5)
        ],
    )

    # Ensure that the type and number of choices are correct.
    assert (
        task_strl.getType() == tetrisched.strl.EXPR_MAX
    ), f"Incorrect type of the root of STRL subtree: {task_strl.getType()}"
    assert task_strl.getNumChildren() == 2, "Incorrect number of children in STRL."
    for child in task_strl.getChildren():
        assert (
            child.getType() == tetrisched.strl.EXPR_WINDOWED_CHOOSE
        ), f"Incorrect type of the child of STRL subtree: {child.getType()}"
        assert (
            child.getNumChildren() == 0
        ), f"Incorrect number of children in STRL: {child.getNumChildren()}"


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
        time_discretization=EventTime(10, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    partitions = Partitions(worker_pools=worker_pools)

    # Construct the STRL expression for the TaskGraph.
    task_strls = {}
    task_graph_strl = scheduler._construct_task_graph_strl(
        current_time=EventTime.zero(),
        task=task_one,
        task_graph=task_graph,
        partitions=partitions,
        task_strls=task_strls,
        placement_times_and_rewards=[
            (EventTime.zero(), 1),
            (EventTime(10, EventTime.Unit.US), 1),
            (EventTime(20, EventTime.Unit.US), 1),
            (EventTime(30, EventTime.Unit.US), 1),
        ],
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
        task_one_strl.getType() == tetrisched.strl.EXPR_WINDOWED_CHOOSE
    ), f"Incorrect type: {task_one_strl.getType()}"
    assert task_one_strl.getNumChildren() == 0, "Incorrect number of children in STRL."
    assert (
        task_two_strl.getType() == tetrisched.strl.EXPR_WINDOWED_CHOOSE
    ), f"Incorrect type: {task_two_strl.getType()}"
    assert task_two_strl.getNumChildren() == 0, "Incorrect number of children in STRL."


def test_two_tasks_correctly_scheduled():
    """Tests that two tasks without dependencies are correctly
    scheduled on a single Worker."""
    # Create two tasks.
    task_1 = create_default_task(
        name="task_1",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(
            resource_vector={Resource(name="Slot", _id="any"): 1}
        ),
    )
    task_2 = create_default_task(
        name="task_2",
        runtime=20,
        deadline=30,
        resource_requirements=Resources(
            resource_vector={Resource(name="Slot", _id="any"): 1}
        ),
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
        time_discretization=EventTime(10, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    task_1_placement = placements.get_placements(task_1)[0]
    task_2_placement = placements.get_placements(task_2)[0]
    assert task_1_placement.is_placed(), "Task 1 was not placed."
    assert task_2_placement.is_placed(), "Task 2 was not placed."

    # Test that the Worker placements are correct.
    assert (
        task_1_placement.worker_id == worker_1.id
    ), "Task 1 was not scheduled on the correct worker."
    assert (
        task_2_placement.worker_id == worker_1.id
    ), "Task 2 was not scheduled on the correct worker."

    # Test that the Placement times are linearly-ordered due to one Slot.
    assert (
        # Either Task 1 runs first.
        task_1_placement.placement_time.time + 10
        == task_2_placement.placement_time.time
    ) or (
        # or Task 2 runs first.
        task_2_placement.placement_time.time + 20
        == task_1_placement.placement_time.time
    ), "Task 1 and Task 2 were not scheduled linearly."


def test_two_tasks_with_multiple_strategies_correctly_scheduled():
    """Tests that two tasks without dependencies are correctly scheduled when
    they have multiple strategies."""
    # Create two tasks.
    task_1 = create_default_task(
        name="task_1",
        profile=WorkProfile(
            name="task_1_work_profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_1"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(10, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_2"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(20, EventTime.Unit.US),
                    ),
                ],
            ),
        ),
        deadline=20,
    )
    task_2 = create_default_task(
        name="task_2",
        profile=WorkProfile(
            name="task_2_work_profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_1"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(15, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="Slot_2"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(20, EventTime.Unit.US),
                    ),
                ],
            ),
        ),
        deadline=20,
    )
    task_graph = TaskGraph(name=task_1.task_graph, tasks={task_1: [], task_2: []})
    workload = Workload.from_task_graphs(task_graphs={task_graph.name: task_graph})

    # Release the tasks.
    task_1.release(EventTime.zero())
    task_2.release(EventTime.zero())

    # Construct the WorkerPools.
    # Create the WorkerPools with two workers and one slot each.
    worker_1 = Worker(
        name="worker_1",
        resources=Resources(resource_vector={Resource(name="Slot_1"): 1}),
    )
    worker_2 = Worker(
        name="worker_2",
        resources=Resources(resource_vector={Resource(name="Slot_2"): 1}),
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1, worker_2])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the scheduler and invoke it at the current time.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(5, EventTime.Unit.US),
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    task_1_placement = placements.get_placements(task_1)[0]
    task_2_placement = placements.get_placements(task_2)[0]
    assert not task_1_placement.is_placed(), "Task 1 was not placed."


def test_two_tasks_dependency_correctly_scheduled():
    """Tests that two tasks with a dependency are correctly scheduled."""
    # Create two tasks.
    task_1 = create_default_task(
        name="task_1",
        runtime=10,
        deadline=30,
        resource_requirements=Resources(
            resource_vector={Resource(name="Slot", _id="any"): 1}
        ),
    )
    task_2 = create_default_task(
        name="task_2",
        runtime=20,
        deadline=30,
        resource_requirements=Resources(
            resource_vector={Resource(name="Slot", _id="any"): 1}
        ),
    )
    task_graph = TaskGraph(name=task_1.task_graph, tasks={task_1: [], task_2: [task_1]})
    workload = Workload.from_task_graphs(task_graphs={task_graph.name: task_graph})

    # Release the tasks.
    task_2.release(EventTime.zero())

    # Construct the WorkerPools.
    worker_1 = Worker(
        name="worker_1", resources=Resources(resource_vector={Resource(name="Slot"): 1})
    )
    worker_pool = WorkerPool(name="worker_pool", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the scheduler and invoke it at the current time.
    scheduler = TetriSchedScheduler(
        time_discretization=EventTime(10, EventTime.Unit.US),
        release_taskgraphs=True,
        enforce_deadlines=True,
    )
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    task_1_placement = placements.get_placements(task_1)[0]
    task_2_placement = placements.get_placements(task_2)[0]
    assert task_1_placement.is_placed(), "Task 1 was not placed."
    assert task_2_placement.is_placed(), "Task 2 was not placed."
    assert (
        task_1_placement.worker_id == worker_1.id
    ), "Task 1 was not scheduled on the correct worker."
    assert (
        task_1_placement.placement_time.time == 20
    ), "Task 1 was not scheduled at the correct time."
    assert (
        task_2_placement.worker_id == worker_1.id
    ), "Task 2 was not scheduled on the correct worker."
    assert (
        task_2_placement.placement_time.time == 0
    ), "Task 2 was not scheduled at the correct time."
