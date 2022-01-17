import uuid
import logging
from copy import copy
from collections import defaultdict
from functools import total_ordering
from typing import Mapping, Optional

import utils


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
            _id (`Optional[str]`): The ID of the resource. Use `any` to
                represent a general resource of the given name.
        """
        if _id is not None and _id != 'any':
            raise ValueError("The acceptable values of _id are None / 'any'")
        self._name = name
        self._id = uuid.uuid4() if _id is None else _id

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    def __str__(self):
        return 'Resource(name={}, id={})'.format(self.name, self.id)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.name, self.id))

    def __copy__(self):
        """Copies self and returns a new instance of Resources that shares
        the same reference to the UUID that backs the original instance."""
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, name=self.name)
        instance._id = self._id
        return instance

    def __deepcopy__(self, memo):
        """Deepcopies self and returns a new instance of Resources that has a
        new instance of the UUID with the same value as the original."""
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, name=self.name)
        instance._id = uuid.UUID(self.id)
        memo[id(self)] = instance
        return instance

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
            if self.id == 'any' or other.id == 'any' or \
                    uuid.UUID(self.id) == uuid.UUID(other.id):
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
        _logger (`Optional[logging.Logger]`): The logger to use to log the
            results of the execution.
        __virtual (`Optional[bool]`): Set to True if the Resources are
            virtual and used to test the effects of a potential allocation.
    """
    def __init__(self,
                 resource_vector: Optional[Mapping[Resource, int]] = {},
                 _logger: Optional[logging.Logger] = None,
                 __virtual: Optional[bool] = False):
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        self._resource_vector = defaultdict(int)
        self.__total_resources = defaultdict(int)
        for resource, quantity in resource_vector.items():
            self._resource_vector[copy(resource)] = quantity
            self.__total_resources[copy(resource)] = quantity
        if not all(map(lambda x: type(x) == Resource, self._resource_vector)):
            raise ValueError("The keys for the resource vector "
                             "should be of type 'Resource'")
        self._current_allocations = defaultdict(list)
        self.__virtual = __virtual

    def add_resource(self, resource: Resource, quantity: Optional[int] = 1):
        """Add the given quantity of the specified resource.

        Args:
            resource (`Resource`): The type of the resource to be added.
            quantity (`int`) [default = 1]: The number of the resources to be
                added.
        """
        if type(resource) != Resource:
            raise ValueError("Invalid type for resource: {}".format(
                type(resource)))
        self._resource_vector[resource] += quantity
        self.__total_resources[resource] += quantity
        if not self.__virtual:
            self._logger.debug("Added {} [quantity={}] to {}".format(
                resource, quantity, self))

    def allocate(
            self,
            resource: Resource,
            task: 'Task',  # noqa: F821
            quantity: int = 1):
        """Allocates the given quantity of the specified resource for a
        particular task.

        Args:
            resource (`Resource`): The resource to be allocated.
            task (`Task`): The task to which the resources are allocated.
            quantity (`int`): The amount of the resource to be allocated
                (default = 1).

        Raises:
            `ValueError` if more than the available quantity of the resource is
            requested.
        """
        available_quantity = self.get_available_quantity(resource)
        if available_quantity < quantity:
            raise ValueError("Trying to allocate more than available units of "
                             "{}: requested {}, available {}".format(
                                 resource, quantity, available_quantity))

        # Go over the list of resources and allocate the required number of
        # resources of the given type.
        remaining_quantity = quantity
        for _resource in self._resource_vector:
            if _resource == resource:
                _quantity = self._resource_vector[_resource]
                if _quantity >= remaining_quantity:
                    if not self.__virtual:
                        self._logger.debug(
                            "Allocated {} [quantity={}] from {}".format(
                                _resource, remaining_quantity, self))
                    self._resource_vector[_resource] = (_quantity -
                                                        remaining_quantity)
                    self._current_allocations[task].append(
                        (_resource, remaining_quantity))
                    break
                else:
                    if not self.__virtual:
                        self._logger.debug(
                            "Allocated {} [quantity={}] from {}".format(
                                _resource, _quantity, self))
                    self._resource_vector[_resource] = 0
                    self._current_allocations[task].append(
                        (_resource, _quantity))
                remaining_quantity -= _quantity

            if remaining_quantity == 0:
                break

    def allocate_multiple(
            self,
            resources: 'Resources',
            task: 'Task'  # noqa: F821
    ):
        """Allocates multiple resources together according to their specified
        quantity.

        Args:
            resources (`Resources`): A representation of the `Resources` to be
                allocated.
            task (`Task`): The task to which the resources are allocated.

        Raises:
            `ValueError` if more than the available quantity of any resource
            is requested.
        """
        # Check that all the resources can be allocated.
        for resource, quantity in resources._resource_vector.items():
            available_quantity = self.get_available_quantity(resource)
            if quantity > available_quantity:
                raise ValueError(
                    "Trying to allocate more than the available "
                    "units of {}: requested {}, available {}".format(
                        resource, quantity, available_quantity))

        # Allocate all the resources together.
        for resource, quantity in resources._resource_vector.items():
            if not self.__virtual:
                self._logger.debug("Allocating {} of {} from {} to {}".format(
                    quantity, resource, self, task))
            self.allocate(resource, task, quantity)

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

    def get_allocated_quantity(self, resource: Resource) -> int:
        """Get the quantity of the given `resource` that has been allocated.

        Args:
            resource (`Resource`): The resource whose allocated quantity needs
                to be computed.

        Returns:
            An `int` quantity of the `resource` that has been allocated.
        """
        total_quantity = 0
        for _resource, _quantity in self.__total_resources.items():
            if _resource == resource:
                total_quantity += _quantity
        available_quantity = self.get_available_quantity(resource)
        return total_quantity - available_quantity

    def deallocate(
            self,
            task: 'Task'  # noqa: F821
    ):
        """Deallocates the resources assigned to the particular `task`.

        Args:
            task (`Task`): The task whose assigned resources need to be
                deallocated.

        Raises:
            `ValueError` if the `task` was not allocated from this set of
            Resources.
        """
        if task not in self._current_allocations:
            raise ValueError("The task was not allocated from this Resources.")

        for resource, quantity in self._current_allocations[task]:
            self._resource_vector[resource] += quantity

        del self._current_allocations[task]

    def __str__(self):
        return "Resources({})".format(self._resource_vector)

    def __len__(self):
        return len(self._resource_vector)

    def __copy__(self) -> 'Resources':
        """Copies self, and returns a new instance with a copy of the original
        Resource availabilities.

        This method keeps the current allocations of Resources in self in the
        new instance.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, self._resource_vector, self._logger, True)

        # Copy over the allocations.
        for task, allocations in self._current_allocations.items():
            instance._current_allocations[task].extend(copy(allocations))

        return instance

    def __deepcopy__(self, memo):
        """Copies self, and returns a new instance with a copy of the original
        Resource availabilities.

        This method removes the current allocation of Resources in self in the
        new instance.
        """
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, self._resource_vector, self._logger, True)

        # Undo the allocations.
        for allocations in self._current_allocations.values():
            for resource, quantity in allocations:
                instance.add_resource(resource, quantity)

        memo[id(self)] = instance
        return instance

    def __repr__(self):
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
        for resource, quantity in other._resource_vector.items():
            if self.get_available_quantity(resource) >= quantity:
                pass
            else:
                return False
        return True

    def __add__(self, other: 'Resources') -> 'Resources':
        """Adds the availability and the allocations of the Resources in self
        to the resources in other and returns a new instances of Resources.

        Args:
            other (`Resources`): The resources set to be added to self.

        Returns:
            The sum of the availability and allocation of the two Resources.
        """
        # Add the resource vector from the two resources.
        resource_vector = defaultdict(int)

        for resource, quantity in self._resource_vector.items():
            resource_vector[resource] += quantity

        for resource, quantity in other._resource_vector.items():
            resource_vector[resource] += quantity

        # Add the total resources vector from the two resources.
        total_resources_vector = defaultdict(int)

        for resource, quantity in self.__total_resources.items():
            total_resources_vector[resource] += quantity

        for resource, quantity in other.__total_resources.items():
            total_resources_vector[resource] += quantity

        # Add the allocation vector from the two resources.
        current_allocations = defaultdict(list)

        for task, allocations in self._current_allocations.items():
            current_allocations[task].extend(allocations)

        for task, allocations in other._current_allocations.items():
            current_allocations[task].extend(allocations)

        # Construct a new Resources instance.
        resources = Resources(_logger=self._logger)
        resources._resource_vector = resource_vector
        resources.__total_resources = total_resources_vector
        resources._current_allocations = current_allocations
        return resources
