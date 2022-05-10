import uuid

import pytest

from workload import Job, JobGraph


def test_job_construction():
    """Test that a Job is correctly constructed."""
    job = Job(name="Perception")
    assert job.name == "Perception", "Job has the wrong name."
    try:
        uuid.UUID(job.id)
    except ValueError:
        assert False, "Job has an incorrect ID."


def test_job_equivalence():
    """Test that two Jobs with the same ID are equivalent."""
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    assert job_1 == job_1, "Job is not equivalent to itself."
    assert job_2 == job_2, "Job is not equivalent to itself."
    assert job_1 != job_2, "Job is equivalent to another Job."


def test_empty_jobgraph_construction():
    """Test that an empty JobGraph is correctly constructed."""
    job_graph = JobGraph()
    assert len(job_graph) == 0, "Default JobGraph should have 0 length."


def test_job_addition_with_no_children():
    """Test that the addition of a Job to the graph works."""
    job_graph = JobGraph()
    assert len(job_graph) == 0, "JobGraph has incorrect length."
    job_graph.add_job(Job(name="Planning"))
    assert len(job_graph) == 1, "JobGraph has incorrect length."


def test_job_addition_with_children():
    """Test that the addition of a Job with children to the graph works."""
    job_graph = JobGraph()
    assert len(job_graph) == 0, "JobGraph has incorrect length."
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    job_graph.add_job(job_1, [job_2])
    assert len(job_graph) == 2, "JobGraph has incorrect length."


def test_successful_retrieval_of_children():
    """Test that the correct children are returned by the JobGraph."""
    job_graph = JobGraph()
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    job_graph.add_job(job_1, [job_2])
    children_job_1 = job_graph.get_children(job_1)
    assert len(children_job_1) == 1, "Incorrect number of children."
    assert children_job_1[0] == job_2, "Incorrect child returned."


def test_successful_retrieval_of_parents():
    """Test that the correct parents are returned by the JobGraph."""
    job_graph = JobGraph()
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    job_graph.add_job(job_1, [job_2])
    parents_job_2 = job_graph.get_parents(job_2)
    assert len(parents_job_2) == 1, "Incorrect number of parents."
    assert parents_job_2[0] == job_1, "Incorrect parent returned."


def test_failed_retrieval_of_children():
    """Test that retrieving children of a non-existent job fails."""
    job_graph = JobGraph()
    job = Job(name="Perception")
    with pytest.raises(ValueError):
        job_graph.get_children(job)


def test_addition_of_child_to_job():
    """Test the successful addition of a child to a job."""
    job_graph = JobGraph()
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    job_graph.add_job(job_1, [])
    assert len(job_graph) == 1, "JobGraph has incorrect lengh."
    job_graph.add_child(job_1, job_2)
    assert len(job_graph) == 2, "JobGraph has incorrect lengh."
    children_job_1 = job_graph.get_children(job_1)
    assert len(children_job_1) == 1, "Incorrect number of children."
    assert children_job_1[0] == job_2, "Incorrect child returned."


def test_failed_addition_of_child_to_job():
    """Test that addition of a child to a non-existent job fails."""
    job_graph = JobGraph()
    job_1, job_2 = Job(name="Perception"), Job(name="Planning")
    with pytest.raises(ValueError):
        job_graph.add_child(job_1, job_2)


def test_iteration_jobgraph():
    """Test that the iteration of the JobGraph is in a BFS manner."""
    job_graph = JobGraph()
    camera_job, lidar_job = Job(name="Camera"), Job(name="Lidar")
    lidar_coordinate_mapping_job = Job(name="LidarToCameraMapping")
    perception_job = Job(name="Perception")
    job_graph.add_job(camera_job, [perception_job])
    job_graph.add_job(lidar_job, [lidar_coordinate_mapping_job])
    job_graph.add_job(lidar_coordinate_mapping_job, [perception_job])

    traversal = []
    for job in job_graph:
        traversal.append(job)

    assert traversal == [
        camera_job,
        lidar_job,
        lidar_coordinate_mapping_job,
        perception_job,
    ], "Incorrect BFS traversal returned by the JobGraph."
