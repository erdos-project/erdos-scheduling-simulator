import uuid
from copy import copy, deepcopy

import pytest

from tests.utils import create_default_task
from workload import Resource, Resources


def test_resource_construction():
    """Test that the Resource is constructed with the correct name and ID."""
    cpu_resource = Resource(name="CPU")
    assert cpu_resource.name == "CPU", "Incorrect name for Resource."
    try:
        uuid.UUID(cpu_resource.id)
    except ValueError:
        assert False, "Incorrect ID for Resource."


def test_generic_resource_construction():
    """Test that a Resource with 'any' ID can be constructed."""
    cpu_resource = Resource(name="CPU", _id="any")
    assert cpu_resource.name == "CPU", "Incorrect name for Resource."
    assert cpu_resource.id == "any", "Incorrect ID for Resource."


def test_failure_generic_resource_construction():
    """Test that a Resource with an incorrect ID cannot be constructed."""
    with pytest.raises(ValueError):
        Resource(name="CPU", _id="other")


def test_equivalence_with_self():
    """Test that a Resource is equivalent to itself."""
    cpu_resource = Resource(name="CPU")
    gpu_resource = Resource(name="GPU", _id="any")
    assert cpu_resource == cpu_resource, "CPU Resources are not equivalent."
    assert gpu_resource == gpu_resource, "GPU Resources are not equivalent."
    assert cpu_resource != gpu_resource, "CPU and GPU resources are equivalent."


def test_equivalence_with_any():
    """Test that a particular instance of a Resource is equivalent to any."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU", _id="any")
    assert cpu_resource_1 == cpu_resource_2, "The two CPU resources are not equivalent."


def test_non_equivalence_of_same_type():
    """Test that resources with same name but diff IDs are not equivalent."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    assert (
        cpu_resource_1 != cpu_resource_2
    ), "Two resources with same name but different IDs are equivalent."


def test_resource_hash_equivalence():
    """Test that two similar Resources have similar hash values."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU", _id="any")
    cpu_resource_3 = Resource(name="CPU", _id="any")

    assert hash(cpu_resource_1) != hash(
        cpu_resource_2
    ), "The hash values of different Resources are not similar."

    assert hash(cpu_resource_2) == hash(
        cpu_resource_3
    ), "The hash values of similar Resources are not similar."


def test_resource_copy():
    """Test that copying a Resource works correctly."""
    cpu_resource = Resource(name="CPU")
    cpu_resource_copy = copy(cpu_resource)
    assert id(cpu_resource) != id(
        cpu_resource_copy
    ), "The copy should not have the same ID as the original."
    assert id(cpu_resource._id) == id(
        cpu_resource_copy._id
    ), "The copy should have an identical ID for its identifier."


def test_resource_deepcopy():
    """Test that deep copying a Resource works correctly."""
    cpu_resource = Resource(name="CPU")
    cpu_resource_copy = deepcopy(cpu_resource)
    assert id(cpu_resource) != id(
        cpu_resource_copy
    ), "The deepcopy should not have the same ID as the original."
    assert id(cpu_resource._id) != id(
        cpu_resource_copy._id
    ), "The deepcopy should not have an identical ID for its identifier."


def test_empty_resources_construction():
    """Test that a default initialization of Resources is empty."""
    resources = Resources()
    assert len(resources) == 0, "Non-zero resources in an empty Resources construction."


def test_non_empty_resource_construction():
    """Test that an initialization of Resources with a vector is correct."""
    cpu_resource, gpu_resource = Resource(name="CPU"), Resource(name="GPU")
    resources = Resources(resource_vector={cpu_resource: 1, gpu_resource: 2})
    assert (
        len(resources) == 2
    ), "The number of resources do not match the resource vector."


def test_addition_of_resource():
    """Test that resources are added correctly using add_resource()."""
    resources = Resources()
    assert len(resources) == 0, "Non-zero resources in an empty Resources construction."
    resources.add_resource(Resource(name="CPU"))
    assert len(resources) == 1, "Resources were not added correctly."


def test_failed_addition_of_resource():
    """Test that resources are correctly type checked in add_resource()."""
    resources = Resources()
    assert len(resources) == 0, "Non-zero resources in an empty Resources construction."
    with pytest.raises(ValueError):
        resources.add_resource("CPU")


def test_correct_quantity():
    """Test that the correct quantity is returned by get_quantity."""
    cpu_resource = Resource(name="CPU")
    resources = Resources(resource_vector={cpu_resource: 10})
    assert (
        resources.get_available_quantity(cpu_resource) == 10
    ), "Incorrect quantity of CPU resource retrieved."


def test_correct_quantity_for_any_available_resource():
    """Test that the correct quantity is returned by get_quantity."""
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources(resource_vector={Resource(name="CPU"): 10})
    assert (
        resources.get_available_quantity(cpu_resource_any) == 10
    ), "Incorrect quantity of CPU resource retrieved."


def test_extension_of_resource():
    """Test that resources are added correctly using add_resource()."""
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    assert (
        resources.get_available_quantity(cpu_resource_any) == 10
    ), "Incorrect quantity of the CPU resource."
    resources.add_resource(Resource(name="CPU", _id="any"), quantity=10)
    assert (
        resources.get_available_quantity(cpu_resource_any) == 20
    ), "Incorrect quantity of the CPU resource."


def test_allocation_of_resources():
    """Test that resources are correctly allocated."""
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    assert (
        resources.get_available_quantity(cpu_resource_any) == 10
    ), "Incorrect quantity of the CPU resource."
    resources.allocate(cpu_resource_any, create_default_task(), 5)
    assert (
        resources.get_available_quantity(cpu_resource_any) == 5
    ), "Incorrect quantity of the CPU resource."


def test_failed_allocation_of_resources():
    """Test that incorrectly allocating resources fails."""
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_any: 10})
    with pytest.raises(ValueError):
        resources.allocate(cpu_resource_any, create_default_task(), 20)


def test_allocation_of_resources_complex():
    """Test that a complex allocation of resources is correct."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    cpu_resource_any = Resource(name="CPU", _id="any")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5})
    assert (
        resources.get_available_quantity(cpu_resource_any) == 10
    ), "Incorrect quantity of the CPU resource."
    resources.allocate(cpu_resource_any, create_default_task(), 6)
    assert (
        resources.get_available_quantity(cpu_resource_any) == 4
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(cpu_resource_1) == 0
        or resources.get_available_quantity(cpu_resource_2) == 0
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(cpu_resource_1) == 4
        or resources.get_available_quantity(cpu_resource_2) == 4
    ), "Incorrect quantity of the CPU resource."


def test_allocation_of_multiple_resources():
    """Test that multiple different resources are allocated correctly."""
    cpu_resource, gpu_resource = Resource(name="CPU"), Resource(name="GPU")
    resources = Resources({cpu_resource: 10, gpu_resource: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    assert (
        resources.get_available_quantity(cpu_resource_any) == 10
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(gpu_resource_any) == 10
    ), "Incorrect quantity of the CPU resource."

    resources.allocate_multiple(
        Resources({cpu_resource_any: 5, gpu_resource_any: 5}), create_default_task()
    )

    assert (
        resources.get_available_quantity(cpu_resource_any) == 5
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(gpu_resource_any) == 5
    ), "Incorrect quantity of the CPU resource."


def test_correct_maintenance_of_allocated_resources():
    """Test that Resources correctly maintains the current allocations."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    task = create_default_task()
    resources.allocate(cpu_resource_any, task, 8)
    resources.allocate(gpu_resource_any, task, 5)

    assert (
        resources.get_available_quantity(cpu_resource_any) == 2
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(cpu_resource_1) == 0
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(cpu_resource_2) == 2
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(gpu_resource_1) == 5
    ), "Incorrect quantity of the GPU resource."

    assert len(resources._current_allocations) == 1, "Incorrect number of allocations."
    assert (
        len(resources._current_allocations[task]) == 3
    ), "Incorrect current allocations for task."
    assert resources._current_allocations[task][0] == (
        cpu_resource_1,
        5,
    ), "Incorrect allocation of the CPU resource to task."
    assert resources._current_allocations[task][1] == (
        cpu_resource_2,
        3,
    ), "Incorrect allocation of the CPU resource to task."
    assert resources._current_allocations[task][2] == (
        gpu_resource_1,
        5,
    ), "Incorrect allocation of the GPU resource to task."


def test_deallocation_of_resources():
    """Test that Resources are correctly deallocated."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    task = create_default_task()
    resources.allocate(cpu_resource_any, task, 8)
    resources.allocate(gpu_resource_any, task, 5)

    assert len(resources._current_allocations) == 1, "Incorrect number of allocations."
    assert (
        len(resources._current_allocations[task]) == 3
    ), "Incorrect current allocations for task."

    # Deallocate the resources for the given task.
    resources.deallocate(task)
    assert len(resources._current_allocations) == 0, "Incorrect number of allocations."

    # Check that the Resource quantities are the same.
    assert (
        resources.get_available_quantity(cpu_resource_1) == 5
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(cpu_resource_2) == 5
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources.get_available_quantity(gpu_resource_1) == 10
    ), "Incorrect quantity of the GPU resource."


def test_resources_copy():
    """Test that Resources are correctly copied."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    task = create_default_task()
    resources.allocate(cpu_resource_any, task, 8)
    resources.allocate(gpu_resource_any, task, 5)

    assert len(resources._current_allocations) == 1, "Incorrect number of allocations."
    assert (
        len(resources._current_allocations[task]) == 3
    ), "Incorrect current allocations for task."

    # Copy the resources.
    resources_copy = copy(resources)
    assert (
        len(resources_copy._current_allocations) == 1
    ), "Incorrect number of allocations in the copied Resources."
    assert (
        len(resources_copy._current_allocations[task]) == 3
    ), "Incorrect current allocations for task in the copied Resources."
    assert (
        resources_copy.get_available_quantity(cpu_resource_1) == 0
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources_copy.get_available_quantity(cpu_resource_2) == 2
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources_copy.get_available_quantity(gpu_resource_1) == 5
    ), "Incorrect quantity of the GPU resource."


def test_resources_deepcopy():
    """Test that Resources are correctly copied."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    task = create_default_task()
    resources.allocate(cpu_resource_any, task, 8)
    resources.allocate(gpu_resource_any, task, 5)

    assert len(resources._current_allocations) == 1, "Incorrect number of allocations."
    assert (
        len(resources._current_allocations[task]) == 3
    ), "Incorrect current allocations for task."

    # Copy the resources.
    resources_copy = deepcopy(resources)
    assert (
        resources_copy.get_available_quantity(cpu_resource_1) == 5
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources_copy.get_available_quantity(cpu_resource_2) == 5
    ), "Incorrect quantity of the CPU resource."
    assert (
        resources_copy.get_available_quantity(gpu_resource_1) == 10
    ), "Incorrect quantity of the GPU resource."


def test_resources_get_allocated_quantity():
    """Test that get_allocated_quantity() method of Resources works."""
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    cpu_resource_any = Resource(name="CPU", _id="any")
    gpu_resource_any = Resource(name="GPU", _id="any")

    task = create_default_task()
    resources.allocate(cpu_resource_any, task, 8)

    assert (
        resources.get_available_quantity(cpu_resource_any) == 2
    ), "Incorrect quantity of available CPU resources."
    assert (
        resources.get_available_quantity(gpu_resource_any) == 10
    ), "Incorrect quantity of available GPU resources."

    assert (
        resources.get_allocated_quantity(cpu_resource_any) == 8
    ), "Incorrect quantity of allocated CPU resources."
    assert (
        resources.get_allocated_quantity(gpu_resource_any) == 0
    ), "Incorrect quantity of allocated GPU resources."


def test_resources_addition():
    """Test that the addition of two Resources works correctly."""
    # Construct the first set of Resources, and allocate some to a task.
    cpu_resource_1 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources_1 = Resources({cpu_resource_1: 10, gpu_resource_1: 10})
    task_1 = create_default_task()
    resources_1.allocate(cpu_resource_1, task_1, 5)

    # Construct the second set of Resources, and allocate some to a task.
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_2 = Resource(name="GPU")
    resources_2 = Resources({cpu_resource_2: 10, gpu_resource_2: 10})
    task_2 = create_default_task()
    resources_2.allocate(gpu_resource_2, task_2, 5)

    assert (
        resources_1.get_available_quantity(cpu_resource_1) == 5
    ), "Incorrect quantity of available CPU resources."
    assert (
        resources_2.get_available_quantity(cpu_resource_2) == 10
    ), "Incorrect quantity of available CPU resources."

    assert (
        resources_1.get_available_quantity(gpu_resource_1) == 10
    ), "Incorrect quantity of available GPU resources."
    assert (
        resources_2.get_available_quantity(gpu_resource_2) == 5
    ), "Incorrect quantity of available GPU resources."

    # Add the two resources.
    final_resources = resources_1 + resources_2
    assert (
        final_resources.get_available_quantity(cpu_resource_1) == 5
    ), "Incorrect quantity of available CPU resources."
    assert (
        final_resources.get_available_quantity(cpu_resource_2) == 10
    ), "Incorrect quantity of available CPU resources."
    assert (
        final_resources.get_available_quantity(gpu_resource_1) == 10
    ), "Incorrect quantity of available GPU resources."
    assert (
        final_resources.get_available_quantity(gpu_resource_2) == 5
    ), "Incorrect quantity of available GPU resources."

    assert (
        final_resources.get_allocated_quantity(cpu_resource_1) == 5
    ), "Incorrect quantity of allocated CPU resources."
    assert (
        final_resources.get_allocated_quantity(gpu_resource_2) == 5
    ), "Incorrect quantity of allocated GPU resources."


def test_resources_iteration():
    """Test that the iteration over the Resources maintains the correct order."""
    # Construct the first set of Resources.
    cpu_resource_1 = Resource(name="CPU")
    cpu_resource_2 = Resource(name="CPU")
    gpu_resource_1 = Resource(name="GPU")
    resources = Resources({cpu_resource_1: 5, cpu_resource_2: 5, gpu_resource_1: 10})

    # Ensure that the order of the iteration is correct.
    order = [cpu_resource_1, cpu_resource_2, gpu_resource_1]
    for index, resource in enumerate(resources):
        assert order[index] == resource, "The wrong item was yielded by the iterator."

    # Add a new Resource and ensure that the order is still maintained.
    gpu_resource_2 = Resource(name="GPU")
    resources.add_resource(gpu_resource_2, 5)
    order.append(gpu_resource_2)
    for index, resource in enumerate(resources):
        assert order[index] == resource, "The wrong item was yielded by the iterator."
