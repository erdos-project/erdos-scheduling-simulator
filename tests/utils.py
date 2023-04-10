from utils import EventTime
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    Resource,
    Resources,
    Task,
    WorkProfile,
)


def create_default_task(
    name=None,
    job=Job(
        name="Perception",
        profile=WorkProfile(
            name="Perception_Work_Profile",
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
    task_name = name if name else f"{job.name}_Task"
    return Task(
        name=task_name,
        task_graph=task_graph_name,
        job=job,
        profile=WorkProfile(
            name=f"{task_name}_work_profile",
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=resource_requirements,
                        batch_size=1,
                        runtime=EventTime(runtime, EventTime.Unit.US),
                    )
                ]
            ),
        ),
        deadline=EventTime(deadline, EventTime.Unit.US),
        timestamp=timestamp,
        release_time=EventTime(release_time, EventTime.Unit.US),
        start_time=EventTime(start_time, EventTime.Unit.US),
        completion_time=EventTime(completion_time, EventTime.Unit.US),
    )
