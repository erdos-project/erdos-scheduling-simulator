import uuid
from typing import Mapping, Optional, Sequence


class Resource(object):
    """A `Resource` object is used to represent a particular resource in the
    system, and contains metadata about if its available or allocated to a
    particular task.
    """
    def __init__(self, name: str):
        self._name = name
        self._id = uuid.uuid4()
        self._assigned = None

    def assign(self, task_id: str):
        """Assign this Resource instance to the given task_id.

        Args:
            task_id (`str`): The task ID to assign this Resource to.
        """
        self._assigned = task_id

    def is_of_type(self, resource_str: str) -> bool:
        """Checks if the resource is of the given type.

        The resource_str is a representation of the Resource consisting of the
        'name:id', with the id being possibly empty, representing any resource
        of the type 'name'.

        Args:
            resource_str (`str`): A representation of the Resource.

        Returns:
           A boolean representing whether this Resource is of the particular
           type.
        """
        resource_split = resource_str.split(':', 1)
        if len(resource_split) == 1:
            if resource_split[0] == self.name:
                return True
        else:
            _name, _id = resource_split
            if _name == self.name and _id == self.id:
                return True
        return False

    def get_type(self) -> str:
        """Retrieve a textual representation of the Resource.

        The representation is of the form 'name:uuid'.

        Returns:
            A textual representation of the Resource.
        """
        return '{}:{}'.format(self.name, self.id)

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def is_assigned(self):
        return self._assigned is None

    def __str__(self):
        return 'Resource(name={}, id={}, assigned={})'.format(self.name,
                                                              self.id,
                                                              self._assigned)

    def __repr__(self):
        return str(self)


class Resources(object):
    """A `Resources` object is used to represent the availability / request of
    a particular set of resources. It can be used by a `Worker` or a
    `WorkerPool` to represent the set of available resources, and by a `Task`
    to represent the set of required resources.

    This object contains a mapping from an arbitrary resource (specified by a
    `str`) to its available / requested quantity, and provides general methods
    that may be useful to a simulator to check if resource requests can be
    fulfilled.

    Args:
        resource_vector (Mapping[str, int]): A mapping from an arbitrary set of
            resources to their quantities.
    """
    def __init__(self, resource_vector: Optional[Sequence[Resource]]):
        self._resource_vector = [] if resource_vector is None else\
                resource_vector

    def add_resource(self, resource_name: str, quantity: Optional[int] = 1):
        """Add the given quantity of the specified resource.

        Args:
            resource_name (`str`): The type of the resource to be added.
            quantity (`int`) [default = 1]: The number of the resources to be
                added.
        """
        self._resource_vector.extend([Resource(name=resource_name)
                                      for _ in range(quantity)])

    def get_available_quantity(self, resource: str) -> int:
        """Provides the quantity of the available resources of the given type.

        If the resource is of type 'name:uuid', then the quantity of that
        specific resource is returned. Otherwise, a sum of all the quantities
        of a resource of type 'name' is returned.

        Args:
            resource (`str`): The resource to retrieve the available quantity
                of.

        Returns:
            The available quantity of the given resource.
        """
        resource_quantity = 0
        for _resource in self._resource_vector:
            if _resource.is_of_type(resource):
                resource_quantity += 1
        return resource_quantity

    def allocate(self, resource: str, task_id: str, quantity: int = 1):
        """Allocates the given quantity of the specified resource for a
        particular task.

        Args:
            resource (`str`): The resource to be allocated.
            task_id  (`str`): The ID of the Task to assign the Resource to.
            quantity (`int`): The amount of the resource to be allocated
                (default = 1).

        Raises:
            `ValueError` if more than the available quantity of the resource is
            requested.
        """
        available_quantity = self.get_available_quantity(resource)
        if available_quantity < quantity:
            raise ValueError("Trying to allocate more than available units of \
                             {}: requested {}, available {}".format(resource,
                             quantity, available_quantity))

        # Go over the list of resources and allocate the required number of
        # resources of the given type.
        allocated_quantity = 0
        for _resource in self._resource_vector:
            if _resource.is_of_type(resource):
                _resource.assign(task_id)
                allocated_quantity += 1

            if allocated_quantity == quantity:
                break

    def allocate_multiple(self, resource_vector: Mapping[str, int]):
        """Allocates multiple resources together according to their specified
        quantity.

        Args:
            resource_vector (`Mapping[str, int]`): A mapping of the resource
                type to the required quantity.

        Raises:
            `ValueError` if more than the available quantity of any resource
            is requested.
        """
        # Check that all the resources can be allocated.
        for resource, quantity in resource_vector:
            available_quantity = self.get_available_quantity(resource)
            if quantity > available_quantity:
                raise ValueError("Trying to allocate more than the available \
                        units of {}: requested {}, available {}".format(
                            resource, quantity, available_quantity))

        # Allocate all the resources together.
        for resource, quantity in resource_vector:
            self.allocate(resource, quantity)

    def __str__(self):
        return "Resources({})".format(self._resource_vector)

    def repr(self):
        return str(self)
