import pytest

from workers import Worker, WorkerPool
from workload import Resource, Resources

from tests.test_tasks import __create_default_task


def test_worker_construction():
    """ Test that a Worker can be constructed successfully. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    assert worker.name == "Worker_1", "Incorrect name for Worker."
    assert worker.num_threads == 2, "Incorrect number of threads for Worker."


def test_worker_task_accomodation():
    """ Test if a Worker's can_accomodate_task works correctly. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    task = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}))
    assert worker.can_accomodate_task(task),\
        "Worker should have been able to accomodate the task."


def test_worker_task_accomodation_fail():
    """ Test that a Worker's can_accomodate_task works correctly. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    task = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 2}))
    assert not worker.can_accomodate_task(task),\
        "Worker should not have been able to accomodate the task."


def test_worker_place_task():
    """ Test that placing a Task correctly accounts for Resources. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    task = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1,
                         Resource(name="GPU", _id="any"): 1, }))

    # Place the task.
    worker.place_task(task)
    placed_tasks = worker.get_placed_tasks()

    assert len(placed_tasks) == 1, "Incorrect number of placed tasks."
    assert placed_tasks[0] == task, "Incorrect placed task."
    assert worker.resources.get_available_quantity(
            Resource(name="CPU", _id="any")) == 0,\
        "Incorrect number of CPU resources available."
    assert worker.resources.get_available_quantity(
            Resource(name="GPU", _id="any")) == 0,\
        "Incorrect number of GPU resources available."


def test_worker_step_tasks():
    """ Test that a Worker correctly steps all of its tasks. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    task_one = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1, }, ),
        runtime=3.0)
    task_two = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 1, }, ),
        runtime=3.0)

    # Place the tasks.
    worker.place_task(task_one)
    worker.place_task(task_two)

    # Release and start the tasks.
    task_one.release(2.0)
    task_one.start(3.0)
    task_two.release(2.0)
    task_two.start(3.0)

    # Step through the Worker's tasks.
    worker.step(3.0)
    worker.step(4.0)
    worker.step(5.0)

    # Ensure that the tasks are finished.
    assert task_one.is_complete(), "Task should have been completed."
    assert task_two.is_complete(), "Task should have been completed."


def test_worker_pool_construction():
    """ Test that a WorkerPool is correctly constructed. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    worker_pool = WorkerPool(name="WorkerPool_Test", workers=[worker])
    assert len(worker_pool.workers) == 1, "Incorrect numher of Workers."


def test_addition_of_workers():
    """ Test that Workers can be correctly added to the WorkerPool. """
    worker = Worker(name="Worker_1",
                    resources=Resources({
                                Resource(name="CPU"): 1,
                                Resource(name="GPU"): 1}),
                    num_threads=2)
    worker_pool = WorkerPool(name="WorkerPool_Test", workers=[worker])
    assert len(worker_pool.workers) == 1, "Incorrect numher of Workers."
    worker_pool.add_workers([
        Worker(name="Worker_1",
               resources=Resources({
                            Resource(name="CPU"): 1,
                            Resource(name="GPU"): 1}),
               num_threads=2),
    ])
    assert len(worker_pool.workers) == 2, "Incorrect number of Workers."


def test_place_task_on_worker_pool():
    """ Test the placement of tasks on a WorkerPool. """
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(name="Worker_1",
                        resources=Resources({
                            Resource(name="CPU"): 1,
                            Resource(name="GPU"): 1}),
                        num_threads=2,
                        )
    worker_two = Worker(name="Worker_2",
                        resources=Resources({
                            Resource(name="CPU"): 1,
                            Resource(name="GPU"): 1}),
                        num_threads=2,
                        )
    worker_pool = WorkerPool(name="WorkerPool_Test",
                             workers=[worker_one, worker_two],
                             )

    # Place a task on the WorkerPool.
    task_one = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}))
    worker_pool.place_task(task_one)

    assert len(worker_pool.get_placed_tasks()) == 1,\
        "Incorrect number of placed tasks."
    assert (len(worker_one.get_placed_tasks()) == 1 and
            len(worker_two.get_placed_tasks()) == 0), "Incorrect placement."

    # Place another task on the WorkerPool.
    task_two = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}))
    worker_pool.place_task(task_two)

    assert len(worker_pool.get_placed_tasks()) == 2,\
        "Incorrect number of placed tasks."
    assert (len(worker_one.get_placed_tasks()) == 1 and
            len(worker_two.get_placed_tasks()) == 1), "Incorrect placement."

    # Fail to place a task on the WorkerPool.
    task_three = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 2}))
    with pytest.raises(ValueError):
        worker_pool.place_task(task_three)

    # Place another task on the WorkerPool.
    task_four = __create_default_task(resource_requirements=Resources(
        resource_vector={Resource(name="GPU", _id="any"): 1}))
    worker_pool.place_task(task_four)

    assert len(worker_pool.get_placed_tasks()) == 3,\
        "Incorrect number of placed tasks."
    assert (len(worker_one.get_placed_tasks()) == 2 and
            len(worker_two.get_placed_tasks()) == 1), "Incorrect placement."


def test_worker_pool_step():
    """ Tests that WorkerPool's step() correctly returns completed tasks. """
    # Initialize the Workers and the WorkerPool.
    worker_one = Worker(name="Worker_1",
                        resources=Resources({
                            Resource(name="CPU"): 1,
                            Resource(name="GPU"): 1}),
                        num_threads=2,
                        )
    worker_two = Worker(name="Worker_2",
                        resources=Resources({
                            Resource(name="CPU"): 1,
                            Resource(name="GPU"): 1}),
                        num_threads=2,
                        )
    worker_pool = WorkerPool(name="WorkerPool_Test",
                             workers=[worker_one, worker_two],
                             )

    # Place a task on the WorkerPool.
    task_one = __create_default_task(
            resource_requirements=Resources(
                    resource_vector={Resource(name="CPU", _id="any"): 1}),
            runtime=3.0,
            )
    worker_pool.place_task(task_one)

    assert len(worker_pool.get_placed_tasks()) == 1,\
        "Incorrect number of placed tasks."

    # Place another task on the WorkerPool.
    task_two = __create_default_task(
            resource_requirements=Resources(
                    resource_vector={Resource(name="CPU", _id="any"): 1}),
            runtime=5.0,
            )
    worker_pool.place_task(task_two)

    assert len(worker_pool.get_placed_tasks()) == 2,\
        "Incorrect number of placed tasks."

    # Release and start the two tasks.
    task_one.release(2.0)
    task_two.release(2.0)
    task_one.start(3.0)
    task_two.start(3.0)

    # Step through the WorkerPool and ensure that the correct completed tasks
    # are returned at the correct simulation time.
    completed_tasks = worker_pool.step(3.0)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(4.0)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(5.0)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_one, "Incorrect completed task."
    completed_tasks = worker_pool.step(6.0)
    assert len(completed_tasks) == 0, "Incorrect number of completed tasks."
    completed_tasks = worker_pool.step(7.0)
    assert len(completed_tasks) == 1, "Incorrect number of completed tasks."
    assert completed_tasks[0] == task_two, "Incorrect completed task."
