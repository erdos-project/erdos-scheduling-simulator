import uuid
import pytest

from workload import Resource


def test_resource_construction():
    """ Test that the Resource is constructed with the correct name and ID. """
    cpu_resource = Resource(name="CPU")
    assert cpu_resource.name == "CPU", "Incorrect name for Resource."
    try:
        uuid.UUID(cpu_resource.id)
    except ValueError:
        assert False, "Incorrect ID for Resource."


def test_generic_resource_construction():
    """ Test that a Resource with 'any' ID can be constructed. """
    cpu_resource = Resource(name="CPU", _id="any")
    assert cpu_resource.name == "CPU", "Incorrect name for Resource."
    assert cpu_resource.id == "any", "Incorrect ID for Resource."


def test_failure_generic_resource_construction():
    """ Test that a Resource with an incorrect ID cannot be constructed. """
    with pytest.raises(ValueError):
        Resource(name="CPU", _id="other")


def test_equivalence_with_self():
    """ Test that a Resource is equivalent to itself. """
    cpu_resource = Resource(name="CPU")
    gpu_resource = Resource(name="GPU", _id="any")
    assert cpu_resource == cpu_resource, "CPU Resources are not equivalent."
    assert gpu_resource == gpu_resource, "GPU Resources are not equivalent."
    assert cpu_resource != gpu_resource,\
        "CPU and GPU resources are equivalent."


def test_equivalence_with_any():
    """ Test that a particular instance of a Resource is equivalent to any. """
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU", _id="any")
    assert cpu_resource_1 == cpu_resource_2,\
        "The two CPU resources are not equivalent."


def test_non_equivalence_of_same_type():
    """ Test that resources with same name but diff IDs are not equivalent. """
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    assert cpu_resource_1 != cpu_resource_2,\
        "Two resources with same name but different IDs are equivalent."


def test_non_task_assignment():
    """ Test that a Resource starts without being an assigned task. """
    cpu_resource = Resource(name="CPU")
    assert not cpu_resource.is_assigned,\
        "Resource is assigned without calling assign()"


def test_failed_task_assignment():
    """ Test that a Resource is assigned to a task correctly. """
    with pytest.raises(ValueError):
        cpu_resource = Resource(name="CPU")
        cpu_resource.assign("random")


def test_successful_task_assignment():
    """ Test that a Resource is assigned to a task correctly. """
    test_uuid = uuid.uuid4()

    # Assign and check first Resource.
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_1.assign(str(test_uuid))
    assert cpu_resource_1.is_assigned, "The Resource was not assigned."

    # Assign and check second Resource.
    cpu_resource_2 = Resource(name="CPU")
    cpu_resource_2.assign(test_uuid)
    assert cpu_resource_2.is_assigned, "The Resource was not assigned."
