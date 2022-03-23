from copy import copy, deepcopy

import pytest

from tests.utils import create_default_task
from workers import Worker, WorkerPool
from workload import Resource, Resources


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
    assert worker.can_accomodate_task(
        task
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
    assert not worker.can_accomodate_task(
        task
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
    worker.place_task(task)
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
    worker.place_task(task)
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
    worker.place_task(task)
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


def test_worker_preempt_task_failed():
    """Test that preempting a non-placed Task raises an error."""
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
        worker.preempt_task(task)


def test_worker_preempt_task_success():
    """Test that preempting a Task correctly maintains the Resources."""
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
    worker.place_task(task)
    assert len(worker.get_placed_tasks()) == 1, "Incorrect number of placed tasks."
    assert (
        worker.resources.get_available_quantity(Resource(name="CPU", _id="any")) == 0
    ), "Incorrect number of CPU resources available."
    assert (
        worker.resources.get_available_quantity(Resource(name="GPU", _id="any")) == 0
    ), "Incorrect number of GPU resources available."

    # Run the task.
    task.release(1)
    task.start(2)
    task.pause(3)

    # Preempt the task and ensure correct resources.
    worker.preempt_task(task)
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
    worker.place_task(task_one)
    worker.place_task(task_two)

    # Release and start the tasks.
    task_one.release(2)
    task_one.start(3)
    task_two.release(2)
    task_two.start(3)

    # Step through the Worker's tasks.
    worker.step(3)
    worker.step(4)
    worker.step(5)

    # Ensure that the tasks are finished.
    assert task_one.is_complete(), "Task should have been completed."
    assert task_two.is_complete(), "Task should have been completed."


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
    worker_pool.place_task(task_one)

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
    worker_pool.place_task(task_two)

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
    with pytest.raises(ValueError):
        worker_pool.place_task(task_three)

    # Place another task on the WorkerPool.
    task_four = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="GPU", _id="any"): 1}
        )
    )
    worker_pool.place_task(task_four)

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
    worker_pool.place_task(task_one)

    assert len(worker_pool.get_placed_tasks()) == 1, "Incorrect number of placed tasks."

    # Place another task on the WorkerPool.
    task_two = create_default_task(
        resource_requirements=Resources(
            resource_vector={Resource(name="CPU", _id="any"): 1}
        ),
        runtime=5,
    )
    worker_pool.place_task(task_two)

    assert len(worker_pool.get_placed_tasks()) == 2, "Incorrect number of placed tasks."

    # Release and start the two tasks.
    task_one.release(2)
    task_two.release(2)
    task_one.start(3)
    task_two.start(3)

    # Step through the WorkerPool and ensure that the correct completed tasks
    # are returned at the correct simulation time.
    completed_tasks = worker_pool.step(3)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(4)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(5)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_one, "Incorrect completed task."
    completed_tasks = worker_pool.step(6)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(7)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_two, "Incorrect completed task."


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
    worker_pool.place_task(task_one)

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
    worker_pool.place_task(task_one)

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
