from utils import EventTime
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Resource,
    Resources,
    Task,
)


def create_default_task(
    name=None,
    job=Job(
        name="Perception",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(1000, EventTime.Unit.US),
                )
            ]
        ),
    ),
    resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}
    ),
    runtime=1,
    deadline=10,
    timestamp=0,
    release_time=-1,
    start_time=-1,
    completion_time=-1,
    task_graph_name="TestTaskGraph",
):
    """Helper function to create a default task."""
    return Task(
        name=name if name else f"{job.name}_Task",
        task_graph=task_graph_name,
        job=job,
        available_execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=resource_requirements,
                    batch_size=1,
                    runtime=EventTime(runtime, EventTime.Unit.US),
                )
            ]
        ),
        deadline=EventTime(deadline, EventTime.Unit.US),
        timestamp=timestamp,
        release_time=EventTime(release_time, EventTime.Unit.US),
        start_time=EventTime(start_time, EventTime.Unit.US),
        completion_time=EventTime(completion_time, EventTime.Unit.US),
    )
