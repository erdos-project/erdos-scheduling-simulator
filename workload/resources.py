from typing import Mapping


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
    def __init__(self, resource_vector: Mapping[str, int]):
        self._resource_vector = resource_vector

    def get_available_quantity(self, resource: str) -> int:
        """Provides the quantity of the available resources of the given type.

        Args:
            resource (`str`): The resource to retrieve the available quantity
                of.

        Returns:
            The available quantity of the given resource.
        """
        return self._resource_vector.get(resource, 0)

    def allocate(self, resource: str, quantity: int = 1):
        """Allocates the given quantity of the specified resource.

        Args:
            resource (`str`): The resource to be allocated.
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
        else:
            self._resource_vector[resource] -= quantity

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
