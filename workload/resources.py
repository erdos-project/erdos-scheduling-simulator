import uuid
from collections import defaultdict
from functools import total_ordering
from typing import Mapping, Optional


class Resource(object):
    """A `Resource` object is used to represent a particular resource in the
    system, and contains metadata about if its available or allocated to a
    particular task.

    Args:
        name (`str`): The name of the resource.
        id (`uuid`): The ID of the resource.
    """
    def __init__(self, name: str, _id: Optional[str] = None):
        """Initialize a Resource.

        Args:
            name (`str`): The name of the resource.
            id (`Optional[str]`): The ID of the resource. Use `any` to
                represent a general resource of the given name.
        """
        if _id is not None and _id != 'any':
            raise ValueError("The acceptable values of _id are None / 'any'")
        self._name = name
        self._id = uuid.uuid4() if _id is None else _id
        self._assigned = None

    def assign(self, task_id: str):
        """Assign this Resource instance to the given task_id.

        Args:
            task_id (`str`): The task ID to assign this Resource to.
        """
        self._assigned = task_id

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

    def __hash__(self):
        return hash(self.get_type())

    def __eq__(self, other: 'Resource'):
        """Checks if the current resource is equal to the other.

        To be equivalent, the Resources must have the same name and IDs, or
        one or both of them must have the 'any' ID.

        Args:
            other (`Resource`): The other resource to compare to.

        Returns:
           A boolean representing whether this Resource is of the particular
           type.
        """
        if self.name == other.name:
            if self.id == 'any' or other.id == 'any' or self.id == other.id:
                return True
        return False


@total_ordering
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
        resource_vector (Mapping[Resource, int]): A mapping from an arbitrary
            set of resources to their quantities.
    """
    def __init__(self, resource_vector: Optional[Mapping[Resource, int]]):
        self._resource_vector = defaultdict(int) if resource_vector is None\
                else resource_vector

    def add_resource(self, resource: Resource, quantity: Optional[int] = 1):
        """Add the given quantity of the specified resource.

        Args:
            resource (`Resource`): The type of the resource to be added.
            quantity (`int`) [default = 1]: The number of the resources to be
                added.
        """
        self._resource_vector[resource] += quantity

    def get_available_quantity(self, resource: Resource) -> int:
        """Provides the quantity of the available resources of the given type.

        If the resource has a specific `id`, then the quantity of that resource
        is returned. Otherwise, a sum of all the quantities of resources with
        the `name` are returned.

        Args:
            resource (`Resource`): The resource to retrieve the available
                quantity of.

        Returns:
            The available quantity of the given resource.
        """
        resource_quantity = 0
        for _resource, _quantity in self._resource_vector.items():
            if _resource == resource:
                resource_quantity += _quantity
        return resource_quantity

    def allocate(self, resource: Resource, quantity: int = 1):
        """Allocates the given quantity of the specified resource for a
        particular task.

        Args:
            resource (`Resource`): The resource to be allocated.
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
        remaining_quantity = quantity
        for _resource in self._resource_vector:
            if _resource == resource:
                _quantity = self._resource_vector[_resource]
                if _quantity >= remaining_quantity:
                    self._resource_vector[_resource] = (_quantity -
                                                        remaining_quantity)
                    break
                else:
                    self._resource_vector[_resource] = 0
                remaining_quantity -= _quantity

            if remaining_quantity == 0:
                break

    def allocate_multiple(self, resources: 'Resources'):
        """Allocates multiple resources together according to their specified
        quantity.

        Args:
            resources (`Resources`): A representation of the `Resources` to be
                allocated.

        Raises:
            `ValueError` if more than the available quantity of any resource
            is requested.
        """
        # Check that all the resources can be allocated.
        for resource, quantity in resources._resource_vector:
            available_quantity = self.get_available_quantity(resource)
            if quantity > available_quantity:
                raise ValueError("Trying to allocate more than the available \
                        units of {}: requested {}, available {}".format(
                            resource, quantity, available_quantity))

        # Allocate all the resources together.
        for resource, quantity in resources._resource_vector:
            self.allocate(resource, quantity)

    def __str__(self):
        return "Resources({})".format(self._resource_vector)

    def repr(self):
        return str(self)

    def __gt__(self, other: 'Resources') -> bool:
        """Checks if the given `Resources` are a subset of the current
        `Resources`.

        This method can be used to ascertain if the other set of resources
        can be allocated from the current set of resources.

        Args:
            other (`Resources`): The set of resources to check if they are a
                subset of the current set.

        Returns:
            `True` if other is a subset of self, `False` otherwise.
        """
        for resource, quantity in other._resource_vector:
            if self.get_available_quantity(resource) >= quantity:
                pass
            else:
                return False
        return True
