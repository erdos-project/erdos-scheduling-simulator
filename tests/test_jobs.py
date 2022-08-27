import uuid

import pytest

from utils import EventTime
from workload import Job, JobGraph
from workload.resources import Resources


def test_job_construction():
    """Test that a Job is correctly constructed."""
    job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    assert job.name == "Perception", "Job has the wrong name."
    assert len(job.resource_requirements) == 1, "Job has wrong resource requirements."
    assert (
        job.resource_requirements[0] == Resources()
    ), "Job has wrong resource requirements."
    try:
        uuid.UUID(job.id)
    except ValueError:
        assert False, "Job has an incorrect ID."


def test_conditional_job_construction():
    """Test that a conditional Job is correctly constructed."""
    job = Job(
        name="Perception", runtime=EventTime(1000, EventTime.Unit.US), conditional=True
    )
    assert job.name == "Perception", "Job has the wrong name."
    assert job.conditional, "Job is not conditional."


def test_job_equivalence():
    """Test that two Jobs with the same ID are equivalent."""
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
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
    job_graph.add_job(Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)))
    assert len(job_graph) == 1, "JobGraph has incorrect length."


def test_job_addition_with_children():
    """Test that the addition of a Job with children to the graph works."""
    job_graph = JobGraph()
    assert len(job_graph) == 0, "JobGraph has incorrect length."
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(job_1, [job_2])
    assert len(job_graph) == 2, "JobGraph has incorrect length."


def test_successful_retrieval_of_children():
    """Test that the correct children are returned by the JobGraph."""
    job_graph = JobGraph()
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(job_1, [job_2])
    children_job_1 = job_graph.get_children(job_1)
    assert len(children_job_1) == 1, "Incorrect number of children."
    assert children_job_1[0] == job_2, "Incorrect child returned."


def test_successful_retrieval_of_parents():
    """Test that the correct parents are returned by the JobGraph."""
    job_graph = JobGraph()
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(job_1, [job_2])
    parents_job_2 = job_graph.get_parents(job_2)
    assert len(parents_job_2) == 1, "Incorrect number of parents."
    assert parents_job_2[0] == job_1, "Incorrect parent returned."


def test_failed_retrieval_of_children():
    """Test that retrieving children of a non-existent job fails."""
    job_graph = JobGraph()
    job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    with pytest.raises(ValueError):
        job_graph.get_children(job)


def test_addition_of_child_to_job():
    """Test the successful addition of a child to a job."""
    job_graph = JobGraph()
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
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
    job_1 = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    with pytest.raises(ValueError):
        job_graph.add_child(job_1, job_2)


def test_iteration_jobgraph():
    """Test that the iteration of the JobGraph is in a BFS manner."""
    job_graph = JobGraph()
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
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


def test_depth_first_jobgraph():
    """Test the DFS traversal over the JobGraph."""
    job_graph = JobGraph()
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(camera_job, [perception_job])
    job_graph.add_job(lidar_job, [lidar_coordinate_mapping_job])
    job_graph.add_job(lidar_coordinate_mapping_job, [perception_job])

    traversal = []
    for job in job_graph.depth_first():
        traversal.append(job)

    assert traversal == [
        camera_job,
        perception_job,
        lidar_job,
        lidar_coordinate_mapping_job,
    ] or traversal == [
        lidar_job,
        lidar_coordinate_mapping_job,
        perception_job,
        camera_job,
    ], "Incorrect DFS traversal returned by the JobGraph."


def test_job_depth():
    """Test that the depth of each Job is correct."""
    job_graph = JobGraph()
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(camera_job, [perception_job])
    job_graph.add_job(lidar_job, [lidar_coordinate_mapping_job])
    job_graph.add_job(lidar_coordinate_mapping_job, [perception_job])

    assert job_graph.get_node_depth(camera_job) == 1, "Incorrect job depth."
    assert job_graph.get_node_depth(lidar_job) == 1, "Incorrect job depth."
    assert (
        job_graph.get_node_depth(lidar_coordinate_mapping_job) == 2
    ), "Incorrect job depth."
    assert job_graph.get_node_depth(perception_job) == 3, "Incorrect job depth."


def test_topological_sort_success():
    """Test that the correct topological sort order of the graph is returned."""
    job_graph = JobGraph()
    job_0 = Job(name="Job 0", runtime=EventTime(1000, EventTime.Unit.US))
    job_1 = Job(name="Job 1", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Job 2", runtime=EventTime(1000, EventTime.Unit.US))
    job_3 = Job(name="Job 3", runtime=EventTime(1000, EventTime.Unit.US))
    job_4 = Job(name="Job 4", runtime=EventTime(1000, EventTime.Unit.US))
    job_5 = Job(name="Job 5", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(job_5, [job_0, job_2])
    job_graph.add_job(job_4, [job_0, job_1])
    job_graph.add_job(job_2, [job_3])
    job_graph.add_job(job_3, [job_1])

    topological_sort = job_graph.topological_sort()
    assert len(topological_sort) == 6, "Wrong length of sorting returned."
    for node in job_graph.get_nodes():
        node_index = topological_sort.index(node)
        for child in job_graph.get_children(node):
            assert (
                topological_sort.index(child) > node_index
            ), f"The edge from {node} to {child} was in the wrong order."


def test_topological_sort_fail():
    """Test that the topological sort method raises an error if the graph is not DAG."""
    job_graph = JobGraph()
    job_0 = Job(name="Job 0", runtime=EventTime(1000, EventTime.Unit.US))
    job_1 = Job(name="Job 1", runtime=EventTime(1000, EventTime.Unit.US))
    job_2 = Job(name="Job 2", runtime=EventTime(1000, EventTime.Unit.US))
    job_3 = Job(name="Job 3", runtime=EventTime(1000, EventTime.Unit.US))
    job_4 = Job(name="Job 4", runtime=EventTime(1000, EventTime.Unit.US))
    job_5 = Job(name="Job 5", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(job_5, [job_0, job_2])
    job_graph.add_job(job_4, [job_0, job_1])
    job_graph.add_job(job_2, [job_3])
    job_graph.add_job(job_3, [job_1, job_5])

    with pytest.raises(RuntimeError):
        job_graph.topological_sort()


def test_longest_path():
    """Test that the correct longest path in the graph is returned."""
    job_graph = JobGraph()
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph.add_job(camera_job, [perception_job])
    job_graph.add_job(lidar_job, [lidar_coordinate_mapping_job])
    job_graph.add_job(lidar_coordinate_mapping_job, [perception_job])

    longest_path = job_graph.get_longest_path()
    assert (
        len(longest_path) == 3
    ), "Incorrect longest path length retrieved from the graph."
    assert longest_path == [
        lidar_job,
        lidar_coordinate_mapping_job,
        perception_job,
    ], "Incorrect longest path retrieved from the graph."


def test_completion_time():
    """Test that the correct completion time is computed for the graph."""
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph = JobGraph(
        jobs={
            camera_job: [perception_job],
            lidar_job: [lidar_coordinate_mapping_job],
            lidar_coordinate_mapping_job: [perception_job],
        }
    )

    assert job_graph.completion_time == EventTime(
        3000, EventTime.Unit.US
    ), "Incorrect completion time returned for the JobGraph."


def test_task_independency_same_level():
    """Test that two independent jobs at the same level are not dependent."""
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    lidar_coordinate_logger_job = Job(
        name="LidarToCameraMappingLogger", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph = JobGraph(
        jobs={
            camera_job: [perception_job],
            lidar_job: [lidar_coordinate_mapping_job],
            lidar_coordinate_mapping_job: [perception_job, lidar_coordinate_logger_job],
        }
    )

    assert (
        job_graph.are_dependent(lidar_coordinate_logger_job, perception_job) is False
    ), "The two jobs are not dependent on each other."
    assert (
        job_graph.are_dependent(perception_job, lidar_coordinate_logger_job) is False
    ), "The two jobs are not dependent on each other."


def test_task_independency_different_level():
    """Test that two independent jobs at different levels are not dependent."""
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    lidar_coordinate_logger_job = Job(
        name="LidarToCameraMappingLogger", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    prediction_job = Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph = JobGraph(
        jobs={
            camera_job: [perception_job],
            lidar_job: [lidar_coordinate_mapping_job],
            lidar_coordinate_mapping_job: [perception_job, lidar_coordinate_logger_job],
            perception_job: [prediction_job],
        }
    )

    assert (
        job_graph.are_dependent(lidar_coordinate_logger_job, prediction_job) is False
    ), "The two jobs are not dependent on each other."
    assert (
        job_graph.are_dependent(prediction_job, lidar_coordinate_logger_job) is False
    ), "The two jobs are not dependent on each other."


def test_task_dependency():
    """Test that two dependent jobs at different levels are dependent."""
    camera_job = Job(name="Camera", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_job = Job(name="Lidar", runtime=EventTime(1000, EventTime.Unit.US))
    lidar_coordinate_mapping_job = Job(
        name="LidarToCameraMapping", runtime=EventTime(1000, EventTime.Unit.US)
    )
    lidar_coordinate_logger_job = Job(
        name="LidarToCameraMappingLogger", runtime=EventTime(1000, EventTime.Unit.US)
    )
    perception_job = Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    prediction_job = Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US))
    job_graph = JobGraph(
        jobs={
            camera_job: [perception_job],
            lidar_job: [lidar_coordinate_mapping_job],
            lidar_coordinate_mapping_job: [perception_job, lidar_coordinate_logger_job],
            perception_job: [prediction_job],
        }
    )

    assert job_graph.are_dependent(
        lidar_job, prediction_job
    ), "The two jobs are not dependent on each other."
    assert job_graph.are_dependent(
        prediction_job, lidar_job
    ), "The two jobs are not dependent on each other."
