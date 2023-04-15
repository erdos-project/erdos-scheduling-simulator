from schedulers import ClockworkScheduler
from schedulers.clockwork_scheduler import Model
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
    WorkProfile,
)


def test_clockwork_scheduler_start_simple():
    """Tests that Clockwork's start method loads the work profiles correctly across
    all the workers."""
    clockwork_scheduler = ClockworkScheduler(runtime=EventTime.zero())
    available_models = [
        WorkProfile(
            name="Model_A",
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 10}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    )
                ]
            ),
        ),
        WorkProfile(
            name="Model_B",
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 20}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    )
                ]
            ),
        ),
    ]
    worker_a = Worker(
        name="Worker_A",
        resources=Resources(resource_vector={Resource(name="RAM"): 100}),
    )
    worker_b = Worker(
        name="Worker_B",
        resources=Resources(resource_vector={Resource(name="RAM"): 100}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker_a, worker_b])

    placements = clockwork_scheduler.start(
        EventTime.zero(),
        available_models,
        WorkerPools([worker_pool]),
    )
    assert len(placements) == 4, "Incorrect number of placements for start()."
    assert (
        len(placements.get_placements(computation=available_models[0])) == 2
    ), "The model should be placed on both workers."
    assert (
        len(placements.get_placements(computation=available_models[1])) == 2
    ), "The model should be placed on both workers."


def test_clockwork_scheduler_start_complex():
    """Tests that Clockwork's start method loads the work profiles correctly across
    all the workers."""
    clockwork_scheduler = ClockworkScheduler(runtime=EventTime.zero())
    available_models = [
        WorkProfile(
            name="Model_A",
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 50}
                        ),
                        batch_size=1,
                        runtime=EventTime(100, EventTime.Unit.US),
                    )
                ]
            ),
        ),
        WorkProfile(
            name="Model_B",
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 30}
                        ),
                        batch_size=1,
                        runtime=EventTime(200, EventTime.Unit.US),
                    )
                ]
            ),
        ),
        WorkProfile(
            name="Model_C",
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 30}
                        ),
                        batch_size=1,
                        runtime=EventTime(300, EventTime.Unit.US),
                    )
                ]
            ),
        ),
    ]
    worker_a = Worker(
        name="Worker_A",
        resources=Resources(resource_vector={Resource(name="RAM"): 200}),
    )
    worker_b = Worker(
        name="Worker_B",
        resources=Resources(resource_vector={Resource(name="RAM"): 100}),
    )
    worker_pool = WorkerPool(name="WorkerPool", workers=[worker_a, worker_b])

    placements = clockwork_scheduler.start(
        EventTime.zero(),
        available_models,
        WorkerPools([worker_pool]),
    )
    assert len(placements) == 5, "Incorrect number of placements for start()."
    assert (
        len(placements.get_placements(computation=available_models[2])) == 1
    ), "The final model should be placed on one worker."


def test_model_request_queue():
    """Test that a `Model` instance correctly maintains the request queue."""
    work_profile = WorkProfile(name="Model_A")
    task_a = create_default_task(
        job=Job(name="Job_A"), deadline=10, profile=work_profile
    )
    task_b = create_default_task(
        job=Job(name="Job_B"), deadline=5, profile=work_profile
    )
    task_c = create_default_task(
        job=Job(name="Job_C"), deadline=1, profile=work_profile
    )

    # Construct an instance of `Model` and add the tasks to the request queue.
    model = Model(work_profile)
    model.add_task(task_a)
    assert model._task_priority_queue == [
        task_a
    ], "Incorrect priority queue for Tasks in the Model."
    model.add_task(task_b)
    assert model._task_priority_queue == [
        task_b,
        task_a,
    ], "Incorrect priority queue for Tasks in the Model."
    model.add_task(task_c)
    assert model._task_priority_queue == [
        task_c,
        task_b,
        task_a,
    ], "Incorrect priority queue for Tasks in the Model."


def test_model_available_strategies():
    """Test that a `Model` instance correctly calculates the available strategies."""
    work_profile = WorkProfile(
        name="Model_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=2,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=4,
                    runtime=EventTime(150, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=8,
                    runtime=EventTime(200, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=16,
                    runtime=EventTime(250, EventTime.Unit.US),
                ),
            ]
        ),
    )
    tasks = [
        create_default_task(
            job=Job(name=f"Job_{i}"), deadline=(195 - 2 * i), profile=work_profile
        )
        for i in range(12)
    ]

    # Construct an instance of `Model` and add the tasks to the request queue.
    model = Model(work_profile)
    for task in tasks:
        model.add_task(task)
    assert (
        model.outstanding_requests == 12
    ), "Incorrect number of requests received by the Model."

    available_strategies = model.get_available_execution_strategies(
        current_time=EventTime.zero()
    )
    assert len(available_strategies) == 2, "Incorrect number of available strategies."
    assert available_strategies[0].batch_size == 4, "Incorrect batch size for strategy."
    assert available_strategies[1].batch_size == 2, "Incorrect batch size for strategy."


def test_model_create_placements():
    """Test that the `Model` instance correctly creates the `Placement` objects."""
    work_profile = WorkProfile(
        name="Model_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=4,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
    )
    tasks = [
        create_default_task(
            job=Job(name=f"Job_{i}"),
            deadline=120 - 2 * i,
            profile=work_profile,
        )
        for i in range(6)
    ]

    # Construct an instance of `Model` and add the tasks to the request queue.
    model = Model(work_profile)
    for task in tasks:
        model.add_task(task)
    assert (
        model.outstanding_requests == 6
    ), "Incorrect number of requests received by the Model."
    assert len(model._tasks) == 6, "Incorrect number of tasks in the Model."

    placements = model.get_placements(
        sim_time=EventTime.zero(),
        strategy=work_profile.execution_strategies[0],
        worker_pool_id="TestWorkerPool",
        worker_id="TestWorker",
    )
    assert len(placements) == 4, "Incorrect number of placements created."
    assert len(model._tasks) == 2, "Incorrect number of tasks in the Model."
    assert (
        len(model._task_priority_queue) == 2
    ), "Incorrect number of tasks in the Model."
    assert all(
        [placements[i].task == tasks[-(i + 1)] for i in range(4)]
    ), "Incorrect tasks returned in the Placement."
    batch_strategy = placements[0].execution_strategy
    assert isinstance(batch_strategy, BatchStrategy), "Incorrect strategy type."
    assert all(
        placement.execution_strategy.id == batch_strategy.id for placement in placements
    ), "Incorrect strategy id."
