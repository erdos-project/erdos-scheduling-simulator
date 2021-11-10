import pytest

from workload import Resource, Resources


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
