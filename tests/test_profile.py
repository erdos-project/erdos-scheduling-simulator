from utils import EventTime
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    JobGraph,
    Resource,
    Resources,
    Workload,
    WorkProfile,
)


def test_get_correct_profiles_from_workload_with_jobgraph():
    work_profile_one = WorkProfile(
        name="WorkProfileOne",
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
    )
    work_profile_two = WorkProfile(
        name="WorkProfileTwo",
        execution_strategies=ExecutionStrategies(
            strategies=[
                ExecutionStrategy(
                    resources=Resources(
                        resource_vector={Resource(name="GPU", _id="any"): 1}
                    ),
                    batch_size=1,
                    runtime=EventTime(1000, EventTime.Unit.US),
                )
            ]
        ),
    )
    assert work_profile_one != work_profile_two, "WorkProfiles should be different."
    job_one = Job(name="JobOne", profile=work_profile_one)
    job_graph_one = JobGraph(name="JobGraphOne", jobs={job_one: []})
    job_two = Job(name="JobTwo", profile=work_profile_two)
    job_graph_two = JobGraph(name="JobGraphTwo", jobs={job_two: []})
    workload = Workload.from_job_graphs(
        job_graphs={
            job_graph_one.name: job_graph_one,
            job_graph_two.name: job_graph_two,
        }
    )
    assert len(workload.job_graphs) == 2, "Workload should have two JobGraphs."

    # Get the WorkProfiles from the Workload.
    assert len(workload.work_profiles) == 2, "Workload should have two WorkProfiles."
    assert (
        work_profile_one in workload.work_profiles
    ), "Workload should have WorkProfileOne."
    assert (
        work_profile_two in workload.work_profiles
    ), "Workload should have WorkProfileTwo."
