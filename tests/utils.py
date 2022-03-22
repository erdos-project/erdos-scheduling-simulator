from workload import Job, Resource, Resources, Task


def create_default_task(
    name=None,
    job=Job(name="Perception"),
    resource_requirements=Resources(
        resource_vector={Resource(name="CPU", _id="any"): 1}),
    runtime=1.0,
    deadline=10.0,
    timestamp=0,
    release_time=-1,
    start_time=-1,
    completion_time=-1,
):
    """ Helper function to create a default task. """
    return Task(name=name if name else f"{job.name}_Task",
                job=job,
                resource_requirements=resource_requirements,
                runtime=runtime,
                deadline=deadline,
                timestamp=timestamp,
                release_time=release_time,
                start_time=start_time,
                completion_time=completion_time)
