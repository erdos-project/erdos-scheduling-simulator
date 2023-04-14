from schedulers import ClockworkScheduler
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
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
