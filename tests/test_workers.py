from copy import copy, deepcopy

import pytest

from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Placement,
    Resource,
    Resources,
    WorkProfile,
)


def test_worker_construction():
    """Test that a Worker can be constructed successfully."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    assert worker.name == "Worker_1", "Incorrect name for Worker."
    assert len(worker.resources) == 2, "Incorrect number of Resources."


def test_worker_task_accomodation():
    """Test if a Worker's can_accomodate_task works correctly."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        )
    )
    assert (
        len(worker.get_compatible_strategies(task.available_execution_strategies)) != 0
    ), "Worker should have been able to accomodate the task."


def test_worker_task_accomodation_fail():
    """Test that a Worker's can_accomodate_task works correctly."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 2}
        )
    )
    assert not worker.can_accomodate_strategy(
        task.available_execution_strategies[0]
    ), "Worker should not have been able to accomodate the task."


def test_worker_place_task():
    """Test that placing a Task correctly accounts for Resources."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        )
    )

    # Place the task.
    worker.place_task(task, task.available_execution_strategies[0])
    placed_tasks = worker.get_placed_tasks()

    assert len(placed_tasks) == 1, "Incorrect number of placed tasks."
    assert placed_tasks[0] == task, "Incorrect placed task."
    assert (
        worker.resources.get_available_quantity(Resource(name="CPU", _id="any")) == 0
    ), "Incorrect number of CPU resources available."
    assert (
        worker.resources.get_available_quantity(Resource(name="GPU", _id="any")) == 0
    ), "Incorrect number of GPU resources available."


def test_worker_copy():
    """Test that a copy of the Worker is created correctly."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        )
    )

    # Place the task.
    worker.place_task(task, task.available_execution_strategies[0])
    placed_tasks = worker.get_placed_tasks()

    assert len(placed_tasks) == 1, "Incorrect number of placed tasks."
    assert placed_tasks[0] == task, "Incorrect placed task."

    # Copy the Worker
    worker_copy = copy(worker)
    assert (
        len(worker_copy.get_placed_tasks()) == 1
    ), "Incorrect number of placed tasks in the copy of the Worker."
    assert (
        worker_copy.resources.get_available_quantity(Resource(name="CPU", _id="any"))
        == 0
    ), "Incorrect number of CPU resources available on the copy of Worker."
    assert (
        worker_copy.resources.get_available_quantity(Resource(name="GPU", _id="any"))
        == 0
    ), "Incorrect number of GPU resources available on the copy of Worker."


def test_worker_deepcopy():
    """Test that a deep copy of the Worker is created correctly."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        )
    )

    # Place the task.
    worker.place_task(task, task.available_execution_strategies[0])
    placed_tasks = worker.get_placed_tasks()

    assert len(placed_tasks) == 1, "Incorrect number of placed tasks."
    assert placed_tasks[0] == task, "Incorrect placed task."

    # Copy the Worker
    worker_copy = deepcopy(worker)
    assert (
        len(worker_copy.get_placed_tasks()) == 0
    ), "Incorrect number of placed tasks in the copy of the Worker."
    assert (
        worker_copy.resources.get_available_quantity(Resource(name="CPU", _id="any"))
        == 1
    ), "Incorrect number of CPU resources available on the copy of Worker."
    assert (
        worker_copy.resources.get_available_quantity(Resource(name="GPU", _id="any"))
        == 1
    ), "Incorrect number of GPU resources available on the copy of Worker."


def test_worker_remove_task_failed():
    """Test that removing a non-placed Task raises an error."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        )
    )

    # Ensure failure.
    with pytest.raises(ValueError):
        worker.remove_task(current_time=EventTime.zero(), task=task)


def test_worker_remove_task_success():
    """Test that removing a Task correctly maintains the Resources."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task = create_default_task(
        resource_requirements=Resources(
            resource_vector={
                Resource(name="CPU", _id="any"): 1,
                Resource(name="GPU", _id="any"): 1,
            }
        )
    )

    # Place the task.
    worker.place_task(task, task.available_execution_strategies[0])
    assert len(worker.get_placed_tasks()) == 1, "Incorrect number of placed tasks."
    assert (
        worker.resources.get_available_quantity(Resource(name="CPU", _id="any")) == 0
    ), "Incorrect number of CPU resources available."
    assert (
        worker.resources.get_available_quantity(Resource(name="GPU", _id="any")) == 0
    ), "Incorrect number of GPU resources available."

    # Run the task.
    task.release(EventTime(1, EventTime.Unit.US))
    task.schedule(
        EventTime(2, EventTime.Unit.US),
        placement=Placement.create_task_placement(
            task=task,
            placement_time=EventTime(2, EventTime.Unit.US),
            worker_pool_id=worker.id,
            execution_strategy=task.available_execution_strategies[0],
        ),
    )
    task.start(EventTime(2, EventTime.Unit.US))
    task.preempt(EventTime(3, EventTime.Unit.US))

    # Remove the task and ensure correct resources.
    worker.remove_task(current_time=EventTime(3, EventTime.Unit.US), task=task)
    assert len(worker.get_placed_tasks()) == 0, "Incorrect number of placed tasks."
    assert (
        worker.resources.get_available_quantity(Resource(name="CPU", _id="any")) == 1
    ), "Incorrect number of CPU resources available."
    assert (
        worker.resources.get_available_quantity(Resource(name="GPU", _id="any")) == 1
    ), "Incorrect number of GPU resources available."


def test_worker_step_tasks():
    """Test that a Worker correctly steps all of its tasks."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    task_one = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=3,
    )
    task_two = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 1}
        ),
        runtime=3,
    )

    # Place the tasks.
    worker.place_task(task_one, task_one.available_execution_strategies[0])
    worker.place_task(task_two, task_two.available_execution_strategies[0])

    # Release and start the tasks.
    task_one.release(EventTime(2, EventTime.Unit.US))
    task_one.schedule(
        EventTime(3, EventTime.Unit.US),
        placement=Placement.create_task_placement(
            task=task_one,
            placement_time=EventTime(3, EventTime.Unit.US),
            worker_pool_id=worker.id,
            execution_strategy=task_one.available_execution_strategies[0],
        ),
    )
    task_one.start(EventTime(3, EventTime.Unit.US))
    task_two.release(EventTime(2, EventTime.Unit.US))
    task_two.schedule(
        EventTime(3, EventTime.Unit.US),
        placement=Placement.create_task_placement(
            task=task_two,
            placement_time=EventTime(3, EventTime.Unit.US),
            worker_pool_id=worker.id,
            execution_strategy=task_two.available_execution_strategies[0],
        ),
    )
    task_two.start(EventTime(3, EventTime.Unit.US))

    # Step through the Worker's tasks.
    worker.step(EventTime(3, EventTime.Unit.US))
    worker.step(EventTime(4, EventTime.Unit.US))
    worker.step(EventTime(5, EventTime.Unit.US))

    # Ensure that the tasks are finished.
    assert (
        task_one.remaining_time == EventTime.zero()
    ), "Task should have been completed."
    assert (
        task_two.remaining_time == EventTime.zero()
    ), "Task should have been completed."


def test_worker_pool_construction():
    """Test that a WorkerPool is correctly constructed."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool_Test", workers=[worker])
    assert len(worker_pool.workers) == 1, "Incorrect numher of Workers."


def test_addition_of_workers():
    """Test that Workers can be correctly added to the WorkerPool."""
    worker = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(name="WorkerPool_Test", workers=[worker])
    assert len(worker_pool.workers) == 1, "Incorrect numher of Workers."
    worker_pool.add_workers(
        [
            Worker(
                name="Worker_1",
                resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
            ),
        ]
    )
    assert len(worker_pool.workers) == 2, "Incorrect number of Workers."


def test_place_task_on_worker_pool():
    """Test the placement of tasks on a WorkerPool."""
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_two = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(
        name="WorkerPool_Test",
        workers=[worker_one, worker_two],
    )

    # Place a task on the WorkerPool.
    task_one = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        )
    )
    worker_pool.place_task(
        task_one, execution_strategy=task_one.available_execution_strategies[0]
    )

    assert len(worker_pool.get_placed_tasks()) == 1, "Incorrect number of placed tasks."
    assert (
        len(worker_one.get_placed_tasks()) == 1
        and len(worker_two.get_placed_tasks()) == 0
    ), "Incorrect placement."

    # Place another task on the WorkerPool.
    task_two = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        )
    )
    worker_pool.place_task(
        task_two, execution_strategy=task_two.available_execution_strategies[0]
    )

    assert len(worker_pool.get_placed_tasks()) == 2, "Incorrect number of placed tasks."
    assert (
        len(worker_one.get_placed_tasks()) == 1
        and len(worker_two.get_placed_tasks()) == 1
    ), "Incorrect placement."

    # Fail to place a task on the WorkerPool.
    task_three = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 2}
        )
    )
    assert (
        worker_pool.place_task(
            task_three, execution_strategy=task_three.available_execution_strategies[0]
        )
        is False
    ), "Task should have not been placed."

    # Place another task on the WorkerPool.
    task_four = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 1}
        )
    )
    worker_pool.place_task(
        task_four, execution_strategy=task_four.available_execution_strategies[0]
    )

    assert len(worker_pool.get_placed_tasks()) == 3, "Incorrect number of placed tasks."
    assert (
        len(worker_one.get_placed_tasks()) == 2
        and len(worker_two.get_placed_tasks()) == 1
    ), "Incorrect placement."


def test_worker_pool_step():
    """Tests that WorkerPool's step() correctly returns completed tasks."""
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_two = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(
        name="WorkerPool_Test",
        workers=[worker_one, worker_two],
    )

    # Place a task on the WorkerPool.
    task_one = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=3,
    )
    worker_pool.place_task(task_one, task_one.available_execution_strategies[0])

    assert len(worker_pool.get_placed_tasks()) == 1, "Incorrect number of placed tasks."

    # Place another task on the WorkerPool.
    task_two = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=5,
    )
    worker_pool.place_task(task_two, task_two.available_execution_strategies[0])

    assert len(worker_pool.get_placed_tasks()) == 2, "Incorrect number of placed tasks."

    # Release and start the two tasks.
    task_one.release(EventTime(2, EventTime.Unit.US))
    task_two.release(EventTime(2, EventTime.Unit.US))
    task_one.schedule(
        EventTime(3, EventTime.Unit.US),
        placement=Placement.create_task_placement(
            task=task_one,
            placement_time=EventTime(3, EventTime.Unit.US),
            worker_pool_id=worker_pool.id,
            execution_strategy=task_one.available_execution_strategies[0],
        ),
    )
    task_two.schedule(
        EventTime(3, EventTime.Unit.US),
        placement=Placement.create_task_placement(
            task=task_two,
            placement_time=EventTime(3, EventTime.Unit.US),
            worker_pool_id=worker_pool.id,
            execution_strategy=task_two.available_execution_strategies[0],
        ),
    )
    task_one.start(EventTime(3, EventTime.Unit.US))
    task_two.start(EventTime(3, EventTime.Unit.US))

    # Step through the WorkerPool and ensure that the correct completed tasks
    # are returned at the correct simulation time.
    time = EventTime(3, EventTime.Unit.US)
    completed_tasks = worker_pool.step(current_time=time)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."

    time = EventTime(4, EventTime.Unit.US)
    completed_tasks = worker_pool.step(current_time=time)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."

    time = EventTime(5, EventTime.Unit.US)
    completed_tasks = worker_pool.step(current_time=time)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_one, "Incorrect completed task."
    worker_pool.remove_task(current_time=time, task=completed_tasks[0])

    time = EventTime(6, EventTime.Unit.US)
    completed_tasks = worker_pool.step(current_time=time)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."

    time = EventTime(7, EventTime.Unit.US)
    completed_tasks = worker_pool.step(current_time=time)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_two, "Incorrect completed task."
    worker_pool.remove_task(current_time=time, task=completed_tasks[0])


def test_copy_worker_pool():
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_two = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(
        name="WorkerPool_Test",
        workers=[worker_one, worker_two],
    )

    # Place a task on the WorkerPool.
    task_one = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=3,
    )
    worker_pool.place_task(task_one, task_one.available_execution_strategies[0])

    assert len(worker_pool.get_placed_tasks()) == 1, "Incorrect number of placed tasks."

    # Copy the WorkerPool.
    worker_pool_copy = copy(worker_pool)
    assert (
        len(worker_pool_copy.get_placed_tasks()) == 1
    ), "Incorrect number of placed tasks."
    assert len(worker_pool_copy.workers) == 2, "Incorrect number of workers."
    assert (
        worker_pool_copy.workers[0].resources.get_available_quantity(
            Resource(name="CPU", _id="any")
        )
        == 0
    ), "Incorrect number of available resources in Worker."
    assert (
        worker_pool_copy.workers[1].resources.get_available_quantity(
            Resource(name="CPU", _id="any")
        )
        == 1
    ), "Incorrect number of available resources in Worker."


def test_deepcopy_worker_pool():
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(
        name="Worker_1",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_two = Worker(
        name="Worker_2",
        resources=Resources({Resource(name="CPU"): 1, Resource(name="GPU"): 1}),
    )
    worker_pool = WorkerPool(
        name="WorkerPool_Test",
        workers=[worker_one, worker_two],
    )

    # Place a task on the WorkerPool.
    task_one = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=3,
    )
    worker_pool.place_task(task_one, task_one.available_execution_strategies[0])

    assert len(worker_pool.get_placed_tasks()) == 1, "Incorrect number of placed tasks."

    # Copy the WorkerPool.
    worker_pool_copy = deepcopy(worker_pool)
    assert (
        len(worker_pool_copy.get_placed_tasks()) == 0
    ), "Incorrect number of placed tasks."
    assert len(worker_pool_copy.workers) == 2, "Incorrect number of workers."
    assert (
        worker_pool_copy.workers[0].resources.get_available_quantity(
            Resource(name="CPU", _id="any")
        )
        == 1
    ), "Incorrect number of available resources in Worker."
    assert (
        worker_pool_copy.workers[1].resources.get_available_quantity(
            Resource(name="CPU", _id="any")
        )
        == 1
    ), "Incorrect number of available resources in Worker."


def test_worker_profile_loading():
    """Test that the Worker can correctly load profiles and make it available at the
    correct time."""
    worker = Worker(
        name="Worker_1",
        resources=Resources(resource_vector={Resource(name="RAM"): 100}),
    )
    work_profile = WorkProfile(
        name="TestWorkProfile",
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 50}
                    ),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
            ]
        ),
    )
    assert worker.can_accomodate_strategy(
        work_profile.loading_strategies.get_fastest_strategy()
    ), "Worker should be able to accomodate the LoadingStrategy."
    worker.load_profile(
        profile=work_profile,
        loading_strategy=work_profile.loading_strategies.get_fastest_strategy(),
    )
    assert worker.is_available(work_profile) == EventTime(
        50, EventTime.Unit.US
    ), "Incorrect remaining time until the WorkProfile is available."

    # Step the Worker and recheck the remaining time.
    worker.step(
        current_time=EventTime.zero(), step_size=EventTime(20, EventTime.Unit.US)
    )
    assert worker.is_available(work_profile) == EventTime(
        30, EventTime.Unit.US
    ), "Incorrect remaining time until the WorkProfile is available."

    # Step the Worker and recheck the remaining time.
    worker.step(
        current_time=EventTime(20, EventTime.Unit.US),
        step_size=EventTime(30, EventTime.Unit.US),
    )
    assert (
        worker.is_available(work_profile) == EventTime.zero()
    ), "Incorrect remaining time until the WorkProfile is available."


def test_worker_profile_eviction():
    """Test that a Worker can correctly evict profiles upon request."""
    worker = Worker(
        name="Worker_1",
        resources=Resources(resource_vector={Resource(name="RAM"): 100}),
    )
    work_profile = WorkProfile(
        name="TestWorkProfile",
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 50}
                    ),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
            ]
        ),
    )

    assert worker.can_accomodate_strategy(
        work_profile.loading_strategies.get_fastest_strategy()
    ), "Worker should be able to accomodate the LoadingStrategy."
    worker.load_profile(
        profile=work_profile,
        loading_strategy=work_profile.loading_strategies.get_fastest_strategy(),
    )
    assert worker.is_available(work_profile) == EventTime(
        50, EventTime.Unit.US
    ), "Incorrect remaining time until the WorkProfile is available."

    worker.evict_profile(profile=work_profile)
    assert (
        worker.is_available(work_profile) == EventTime.invalid()
    ), "The profile was not evicted correctly."
