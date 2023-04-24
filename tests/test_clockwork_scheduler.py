from copy import copy

from schedulers import ClockworkScheduler
from schedulers.clockwork_scheduler import Model, Models
from tests.utils import create_default_task
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    BatchStrategy,
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Placement,
    Resource,
    Resources,
    TaskGraph,
    Workload,
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


def test_model_request_queue_simple():
    """Test that a `Model` instance correctly maintains the request queue."""
    work_profile = WorkProfile(
        name="Model_A",
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(), runtime=EventTime.zero(), batch_size=1
                )
            ]
        ),
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(), runtime=EventTime.zero(), batch_size=1
                )
            ]
        ),
    )
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
    assert all(
        list(map(lambda r: r.task, request_queue)) == [task_a]
        for request_queue in model._request_queues.values()
    ), "Incorrect priority queue for Tasks in the Model."
    model.add_task(task_b)
    assert all(
        list(map(lambda r: r.task, request_queue))
        == [
            task_b,
            task_a,
        ]
        for request_queue in model._request_queues.values()
    ), "Incorrect priority queue for Tasks in the Model."
    model.add_task(task_c)
    assert all(
        list(map(lambda r: r.task, request_queue))
        == [
            task_c,
            task_b,
            task_a,
        ]
        for request_queue in model._request_queues.values()
    ), "Incorrect priority queue for Tasks in the Model."


def test_model_request_queue_complex():
    """Test that the Model correctly maintains queues across different batch sizes."""
    work_profile = WorkProfile(
        name="Model_A",
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(), runtime=EventTime.zero(), batch_size=1
                ),
            ]
        ),
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    runtime=EventTime(50, EventTime.Unit.US),
                    batch_size=1,
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    runtime=EventTime(60, EventTime.Unit.US),
                    batch_size=2,
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    runtime=EventTime(75, EventTime.Unit.US),
                    batch_size=4,
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    runtime=EventTime(80, EventTime.Unit.US),
                    batch_size=8,
                ),
            ]
        ),
    )
    tasks = [
        create_default_task(name=f"Task_{i}", profile=work_profile, deadline=100 - i)
        for i in range(10)
    ]

    # Construct an instance of `Model` and add the tasks to the request queue.
    model = Model(work_profile)
    for task in tasks:
        model.add_task(task)
    assert all(
        list(map(lambda r: r.task, request_queue)) == tasks[::-1]
        for request_queue in model._request_queues.values()
    ), "Incorrect priority queue for Tasks in the Model."

    # Retrieve the available strategies and ensure that the correct strategies are
    # returned.
    available_strategies = model.get_available_execution_strategies(
        current_time=EventTime.zero()
    )
    assert len(available_strategies) == 4, "Incorrect number of strategies returned."
    assert all(
        list(map(lambda r: r.task, request_queue)) == tasks[::-1]
        for request_queue in model._request_queues.values()
    ), "Incorrect priority queue for Tasks in the Model."

    # Retrieve the available strategies and ensure that the correct strategies are
    # returned at a future time.
    available_strategies = model.get_available_execution_strategies(
        current_time=EventTime(21, EventTime.Unit.US)
    )
    assert len(available_strategies) == 3, "Incorrect number of strategies returned."
    assert (
        len(
            model._request_queues[
                work_profile.execution_strategies.get_slowest_strategy()
            ]
        )
        == 0
    ), "Incorrect number of requests left in the queue."
    assert all(
        request.num_strategies == 3
        for request in model._tasks.values()
        if request.deadline >= EventTime(96, EventTime.Unit.US)
    ), "Incorrect number of strategies left for each request."
    assert all(
        request.num_strategies == 2
        for request in model._tasks.values()
        if request.deadline < EventTime(96, EventTime.Unit.US)
    ), "Incorrect number of strategies left for each request."


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
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
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
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
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
        len(model._request_queues[work_profile.execution_strategies[0]]) == 2
    ), "Incorrect number of tasks in the Model."
    assert all(
        [placements[i].task == tasks[-(i + 1)] for i in range(4)]
    ), "Incorrect tasks returned in the Placement."
    batch_strategy = placements[0].execution_strategy
    assert isinstance(batch_strategy, BatchStrategy), "Incorrect strategy type."
    assert all(
        placement.execution_strategy.id == batch_strategy.id for placement in placements
    ), "Incorrect strategy id."


def test_model_least_slack_strategy_preferred():
    """Test that Clockwork prefers the strategy with the least slack."""
    work_profile = WorkProfile(
        name="Model_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=8,
                    runtime=EventTime(150, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=4,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
            ]
        ),
    )
    tasks = [
        create_default_task(
            job=Job(name="Job_A"),
            deadline=120,
            profile=work_profile,
        )
        for _ in range(4)
    ]
    tasks.extend(
        [
            create_default_task(
                job=Job(name="Job_B"), deadline=200, profile=work_profile
            )
            for _ in range(8)
        ]
    )

    # Construct an instance of `Model` and add the tasks to the request queue.
    model = Model(work_profile)
    for task in tasks:
        model.add_task(task)

    # Get the available strategies.
    available_strategies = model.get_available_execution_strategies(
        current_time=EventTime(10, EventTime.Unit.US)
    )
    assert len(available_strategies) == 2, "Incorrect number of available strategies."
    assert (
        available_strategies[0].batch_size == 4
    ), "Incorrect batch size for high priority strategy."
    assert (
        available_strategies[1].batch_size == 8
    ), "Incorrect batch size for low priority strategy."


def test_clockwork_task_placement_success_simple():
    """Tests that the scheduler is able to correctly place tasks."""
    # Construct the `Task` objects.
    work_profile_a = WorkProfile(
        name="Model_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=4,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=8,
                    runtime=EventTime(150, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=16,
                    runtime=EventTime(200, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
            ]
        ),
    )
    work_profile_b = WorkProfile(
        name="Model_B",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=4,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=8,
                    runtime=EventTime(150, EventTime.Unit.US),
                ),
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=16,
                    runtime=EventTime(200, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(),
                    batch_size=1,
                    runtime=EventTime(50, EventTime.Unit.US),
                )
            ]
        ),
    )
    tasks = [
        create_default_task(
            job=Job(name=f"Job_A_{i}"),
            deadline=250 - 2 * i,
            profile=work_profile_a,
        )
        for i in range(12)
    ] + [
        create_default_task(
            job=Job(name=f"Job_B_{i}"),
            deadline=250 - 2 * i,
            profile=work_profile_b,
        )
        for i in range(4)
    ]
    task_graph = TaskGraph(name="TestTaskGraph", tasks={task: [] for task in tasks})
    workload = Workload.from_task_graphs(task_graphs={task_graph.name: task_graph})
    assert len(task_graph) == len(tasks), "Incorrect number of tasks in the graph."

    # Construct the `WorkerPool`.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources(resource_vector={Resource(name="GPU"): 1}),
    )
    worker_1.load_profile(
        profile=work_profile_a,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_1.load_profile(
        profile=work_profile_b,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_pool = WorkerPool(name="WorkerPool_1", workers=[worker_1])
    worker_pools = WorkerPools(worker_pools=[worker_pool])

    # Construct the scheduler.
    scheduler = ClockworkScheduler(runtime=EventTime.zero())

    # With one `Worker`, only one of the `Model` should be executed.
    for task in tasks:
        task.release(EventTime.zero())
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    assert len(placements) in {4, 8}, "Incorrect number of placements created."
    if len(placements) == 4:
        # If the placements are for Model_B, check that the request queues are correct.
        assert all(
            placement.task.profile == work_profile_b for placement in placements
        ), "Incorrect tasks returned in the Placement."
        assert (
            scheduler._models[work_profile_b.id].outstanding_requests == 0
        ), "Incorrect number of requests remaining for the Model."
    else:
        # If the placements are for Model_A, check that the request queues are correct.
        assert all(
            placement.task.profile == work_profile_a for placement in placements
        ), "Incorrect tasks returned in the Placement."
        assert (
            scheduler._models[work_profile_a.id].outstanding_requests == 4
        ), "Incorrect number of requests remaining for the Model."

    # With two `Worker`s, both `Model`s should be executed.
    worker_2 = Worker(
        name="Worker_2",
        resources=Resources(resource_vector={Resource(name="GPU"): 1}),
    )
    worker_2.load_profile(
        profile=work_profile_a,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_2.load_profile(
        profile=work_profile_b,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_pool.add_workers([worker_2])
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    assert len(placements) == 12, "Incorrect number of placements created."
    assert (
        len(
            [
                placement
                for placement in placements
                if placement.task.profile == work_profile_a
            ]
        )
        == 8
    ), "Incorrect number of Model_A tasks returned in the Placement."
    assert (
        len(
            [
                placement
                for placement in placements
                if placement.task.profile == work_profile_b
            ]
        )
        == 4
    ), "Incorrect number of Model_B tasks returned in the Placement."
    assert (
        scheduler._models[work_profile_a.id].outstanding_requests == 4
    ), "Incorrect number of requests remaining for the Model A."
    assert (
        scheduler._models[work_profile_b.id].outstanding_requests == 0
    ), "Incorrect number of requests remaining for the Model B."

    # With three `Worker`s, all requests should be finished.
    worker_3 = Worker(
        name="Worker_3",
        resources=Resources(resource_vector={Resource(name="GPU"): 1}),
    )
    worker_3.load_profile(
        profile=work_profile_a,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_3.load_profile(
        profile=work_profile_b,
        loading_strategy=ExecutionStrategy(
            resources=Resources(), batch_size=1, runtime=EventTime.zero()
        ),
    )
    worker_pool.add_workers([worker_3])
    placements = scheduler.schedule(
        sim_time=EventTime.zero(), workload=workload, worker_pools=worker_pools
    )
    assert len(placements) == 16, "Incorrect number of placements created."
    assert (
        len(
            [
                placement
                for placement in placements
                if placement.task.profile == work_profile_a
            ]
        )
        == 12
    ), "Incorrect number of Model_A tasks returned in the Placement."
    assert (
        len(
            [
                placement
                for placement in placements
                if placement.task.profile == work_profile_b
            ]
        )
        == 4
    ), "Incorrect number of Model_B tasks returned in the Placement."
    assert (
        scheduler._models[work_profile_a.id].outstanding_requests == 0
    ), "Incorrect number of requests remaining for the Model A."
    assert (
        scheduler._models[work_profile_b.id].outstanding_requests == 0
    ), "Incorrect number of requests remaining for the Model B."


def test_clockwork_model_loading_simple():
    """Test that the model loading works correctly in ClockworkScheduler."""
    # Create WorkProfiles.
    work_profile_a = WorkProfile(
        name="WorkProfile_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(resource_vector={Resource(name="GPU"): 1}),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(resource_vector={Resource(name="RAM"): 100}),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    work_profile_b = WorkProfile(
        name="WorkProfile_B",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(resource_vector={Resource(name="GPU"): 1}),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(resource_vector={Resource(name="RAM"): 100}),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    model_a = Model(work_profile_a)
    model_b = Model(work_profile_b)

    # Create Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_2 = Worker(
        name="Worker_2",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_pool = WorkerPool(name="WorkerPool_1", workers=[worker_1, worker_2])
    worker_pools = WorkerPools([worker_pool])
    models = Models(models=[model_a, model_b], worker_pools=worker_pools)

    # Ensure that all the Workers have the same priority at the start.
    assert (
        models._workers[worker_1.id].weight == models._workers[worker_2.id].weight
    ), "Workers should have the same weight."

    # If we add all tasks of `Model_A`, then the model should be highest priority.
    tasks = [
        create_default_task(name=f"Task_{i}", profile=work_profile_a, deadline=500)
        for i in range(10)
    ]
    for task in tasks:
        models.add_task(task)
    models.refresh_priorities(worker_pools=worker_pools)
    prioritized_models = models.get_model_priority(worker=worker_1)
    assert prioritized_models == [
        model_a,
        model_b,
    ], "Incorrect prioritization of the models for Worker 1."
    prioritized_models = models.get_model_priority(worker=worker_2)
    assert prioritized_models == [
        model_a,
        model_b,
    ], "Incorrect prioritization of the models for Worker 2."

    # If we add more demand for Model B now, then the model should be highest priority.
    tasks = [
        create_default_task(name=f"Task_{i}", profile=work_profile_b, deadline=500)
        for i in range(20)
    ]
    for task in tasks:
        models.add_task(task)
    models.refresh_priorities(worker_pools=worker_pools)
    assert (
        model_a._outstanding_execution_demand < model_b._outstanding_execution_demand
    ), "Model B should have more outstanding execution demand."
    assert (
        model_a._outstanding_load_demand < model_b._outstanding_load_demand
    ), "Model B should have more outstanding load demand."
    assert (
        models._model_load_priorities[model_a.id]
        < models._model_load_priorities[model_b.id]
    ), "Model B should have higher priority."
    prioritized_models = models.get_model_priority(worker_1)
    assert prioritized_models == [
        model_b,
        model_a,
    ], "Incorrect prioritization of the models for Worker 1."
    prioritized_models = models.get_model_priority(worker_2)
    assert prioritized_models == [
        model_b,
        model_a,
    ], "Incorrect prioritization of the models for Worker 2."


def test_clockwork_model_loading_complex():
    """Test that the model loading works correctly in ClockworkScheduler."""
    # Create WorkProfiles.
    work_profile_a = WorkProfile(
        name="WorkProfile_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 100}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    work_profile_b = WorkProfile(
        name="WorkProfile_B",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 100}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    model_a = Model(work_profile_a)
    model_b = Model(work_profile_b)

    # Create Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_2 = Worker(
        name="Worker_2",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_pool = WorkerPool(name="WorkerPool_1", workers=[worker_1, worker_2])
    worker_pools = WorkerPools([worker_pool])
    models = Models(models=[model_a, model_b], worker_pools=worker_pools)

    # Load one of the models on each of the Workers.
    worker_1.load_profile(
        profile=work_profile_a,
        loading_strategy=ExecutionStrategy(
            resources=work_profile_a.loading_strategies[0].resources,
            batch_size=1,
            runtime=EventTime.zero(),
        ),
    )
    worker_2.load_profile(
        profile=work_profile_b,
        loading_strategy=ExecutionStrategy(
            resources=work_profile_b.loading_strategies[0].resources,
            batch_size=1,
            runtime=EventTime.zero(),
        ),
    )
    # Step so that the models are available at the Workers.
    worker_1.step(current_time=EventTime.zero())
    worker_2.step(current_time=EventTime.zero())

    # Set the last used time of the models so that they are not evicted.
    model_a._last_used_at_worker[worker_1.id] = EventTime(1, EventTime.Unit.US)
    model_b._last_used_at_worker[worker_2.id] = EventTime(1, EventTime.Unit.US)

    # Send same amount of `Task`s for each of the `Model`s.
    tasks = [
        create_default_task(name=f"Task_{i}", profile=work_profile_a, deadline=500)
        for i in range(10)
    ] + [
        create_default_task(name=f"Task_{i}", profile=work_profile_b, deadline=500)
        for i in range(10)
    ]
    for task in tasks:
        models.add_task(task)

    # Ensure that the system is forced to be in a stable state.
    scheduler = ClockworkScheduler(runtime=EventTime.zero())
    scheduler._models = models
    models.refresh_priorities(worker_pools=copy(worker_pools))

    placements = list(
        sorted(
            scheduler.run_load(
                current_time=EventTime(1, EventTime.Unit.US),
                worker_pools=copy(worker_pools),
            ),
            key=lambda p: p.placement_type,
        )
    )
    assert len(placements) == 0, "No evictions / loads should happen with equal queues."

    # Remove all tasks for model B and ensure that the model is evicted
    # (with reduced capacity).
    for task in tasks[10:]:
        model_b.remove_task(task)
    for worker in models._workers.values():
        worker.capacity = EventTime(1, EventTime.Unit.US)
    assert (
        model_b._outstanding_execution_demand == model_b._outstanding_load_demand == 0.0
    ), "Model B should have no outstanding demand."
    models.refresh_priorities(worker_pools=copy(worker_pools))
    placements = scheduler.run_load(EventTime(2, EventTime.Unit.US), copy(worker_pools))
    assert (
        len(placements) == 2
    ), "Model B should be evicted and Model A should be loaded into Worker 2."
    assert (
        placements[0].placement_type == Placement.PlacementType.EVICT_WORK_PROFILE
        and placements[0].work_profile == work_profile_b
    ), "Model B should be evicted."
    assert (
        placements[1].placement_type == Placement.PlacementType.LOAD_WORK_PROFILE
        and placements[1].work_profile == work_profile_a
    ), "Model A should be loaded."


def test_clockwork_lru_eviction_policy():
    """Test that Clockwork follows the LRU eviction policy."""
    # Create WorkProfiles.
    work_profile_a = WorkProfile(
        name="WorkProfile_A",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 100}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    work_profile_b = WorkProfile(
        name="WorkProfile_B",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                ),
            ]
        ),
        loading_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="RAM", _id="any"): 100}
                    ),
                    batch_size=1,
                    runtime=EventTime(100, EventTime.Unit.US),
                )
            ]
        ),
    )
    model_a = Model(work_profile_a)
    model_b = Model(work_profile_b)

    # Create Workers.
    worker_1 = Worker(
        name="Worker_1",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_2 = Worker(
        name="Worker_2",
        resources=Resources(
            resource_vector={Resource(name="GPU"): 1, Resource(name="RAM"): 100}
        ),
    )
    worker_pool = WorkerPool(name="WorkerPool_1", workers=[worker_1, worker_2])
    worker_pools = WorkerPools([worker_pool])
    models = Models(models=[model_a, model_b], worker_pools=worker_pools)

    # Load one of the models on each of the Workers.
    worker_1.load_profile(
        profile=work_profile_a,
        loading_strategy=ExecutionStrategy(
            resources=work_profile_a.loading_strategies[0].resources,
            batch_size=1,
            runtime=EventTime.zero(),
        ),
    )
    worker_2.load_profile(
        profile=work_profile_b,
        loading_strategy=ExecutionStrategy(
            resources=work_profile_b.loading_strategies[0].resources,
            batch_size=1,
            runtime=EventTime.zero(),
        ),
    )
    # Step so that the models are available at the Workers.
    worker_1.step(current_time=EventTime.zero())
    worker_2.step(current_time=EventTime.zero())

    # Set the last used time of only one of the models to check that it is evicted on
    # the correct Worker.
    model_a._last_used_time = EventTime(1, EventTime.Unit.US)
    models.refresh_priorities(worker_pools=copy(worker_pools))

    scheduler = ClockworkScheduler(runtime=EventTime.zero())
    scheduler._models = models
    placements = list(
        sorted(
            scheduler.run_load(
                current_time=EventTime(2, EventTime.Unit.US),
                worker_pools=copy(worker_pools),
            ),
            key=lambda p: p.placement_type,
        )
    )
    assert len(placements) == 2, "Incorrect number of placements received."
    assert (
        placements[0].placement_type == Placement.PlacementType.EVICT_WORK_PROFILE
    ), "Incorrect placement type."
    assert placements[0].worker_id == worker_2.id, "Incorrect worker ID."
    assert placements[0].work_profile == work_profile_b, "Incorrect work profile."

    assert (
        placements[1].placement_type == Placement.PlacementType.LOAD_WORK_PROFILE
    ), "Incorrect placement type."
    assert placements[1].worker_id == worker_2.id, "Incorrect worker ID."
    assert placements[1].work_profile == work_profile_a, "Incorrect work profile."
