import uuid
import pytest

from workload import Resource, Resources


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


def test_resource_hash_equivalence():
    """ Test that two similar Resources have similar hash values. """
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU", _id="any")
    cpu_resource_3 = Resource(name="CPU", _id="any")

    assert hash(cpu_resource_1) != hash(cpu_resource_2),\
        "The hash values of different Resources are not similar."

    assert hash(cpu_resource_2) == hash(cpu_resource_3),\
        "The hash values of similar Resources are not similar."


def test_empty_resources_construction():
    """ Test that a default initialization of Resources is empty. """
    resources = Resources()
    assert len(resources) == 0,\
        "Non-zero resources in an empty Resources construction."


def test_non_empty_resource_construction():
    """ Test that an initialization of Resources with a vector is correct. """
    cpu_resource, gpu_resource = Resource(name="CPU"), Resource(name="GPU")
    resources = Resources(resource_vector={cpu_resource: 1, gpu_resource: 2})
    assert len(resources) == 2,\
        "The number of resources do not match the resource vector."


def test_addition_of_resource():
    """ Test that resources are added correctly using add_resource(). """
    resources = Resources()
    assert len(resources) == 0,\
        "Non-zero resources in an empty Resources construction."
    resources.add_resource(Resource(name="CPU"))
    assert len(resources) == 1, "Resources were not added correctly."


def test_failed_addition_of_resource():
    """ Test that resources are correctly type checked in add_resource(). """
    resources = Resources()
    assert len(resources) == 0,\
        "Non-zero resources in an empty Resources construction."
    with pytest.raises(ValueError):
        resources.add_resource("CPU")


def test_correct_quantity():
    """ Test that the correct quantity is returned by get_quantity. """
    cpu_resource = Resource(name="CPU")
    resources = Resources(resource_vector={cpu_resource: 10})
    assert resources.get_available_quantity(cpu_resource) == 10,\
        "Incorrect quantity of CPU resource retrieved."


def test_correct_quantity_for_any_available_resource():
    """ Test that the correct quantity is returned by get_quantity. """
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources(resource_vector={Resource(name="CPU"): 10})
    assert resources.get_available_quantity(cpu_resource_any) == 10,\
        "Incorrect quantity of CPU resource retrieved."


def test_extension_of_resource():
    """ Test that resources are added correctly using add_resource(). """
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    assert resources.get_available_quantity(cpu_resource_any) == 10,\
        "Incorrect quantity of the CPU resource."
    resources.add_resource(Resource(name="CPU", _id="any"), quantity=10)
    assert resources.get_available_quantity(cpu_resource_any) == 20,\
        "Incorrect quantity of the CPU resource."


def test_allocation_of_resources():
    """ Test that resources are correctly allocated. """
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    assert resources.get_available_quantity(cpu_resource_any) == 10,\
        "Incorrect quantity of the CPU resource."
    resources.allocate(cpu_resource_any, 5)
    assert resources.get_available_quantity(cpu_resource_any) == 5,\
        "Incorrect quantity of the CPU resource."


def test_failed_allocation_of_resources():
    """ Test that incorrectly allocating resources fails. """
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    with pytest.raises(ValueError):
        resources.allocate(cpu_resource_any, 20)


def test_allocation_of_resources_complex():
    """ Test that a complex allocation of resources is correct. """
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5})
    assert resources.get_available_quantity(cpu_resource_any) == 10,\
        "Incorrect quantity of the CPU resource."
    resources.allocate(cpu_resource_any, 6)
    assert resources.get_available_quantity(cpu_resource_any) == 4,\
        "Incorrect quantity of the CPU resource."
    assert (resources.get_available_quantity(cpu_resource_1) == 0 or
            resources.get_available_quantity(cpu_resource_2) == 0),\
        "Incorrect quantity of the CPU resource."
    assert (resources.get_available_quantity(cpu_resource_1) == 4 or
            resources.get_available_quantity(cpu_resource_2) == 4),\
        "Incorrect quantity of the CPU resource."


def test_allocation_of_multiple_resources():
    """ Test that multiple different resources are allocated correctly. """
    cpu_resource, gpu_resource = Resource(name="CPU"), Resource(name="GPU")
    resources = Resources({cpu_resource: 10, gpu_resource: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    assert resources.get_available_quantity(cpu_resource_any) == 10,\
        "Incorrect quantity of the CPU resource."
    assert resources.get_available_quantity(gpu_resource_any) == 10,\
        "Incorrect quantity of the CPU resource."

    resources.allocate_multiple(Resources({cpu_resource_any: 5,
                                           gpu_resource_any: 5}))

    assert resources.get_available_quantity(cpu_resource_any) == 5,\
        "Incorrect quantity of the CPU resource."
    assert resources.get_available_quantity(gpu_resource_any) == 5,\
        "Incorrect quantity of the CPU resource."
