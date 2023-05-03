import random
import uuid
from typing import Optional


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
        if _id is not None and not isinstance(_id, str):
            raise ValueError("The acceptable values of _id are None / a string.")
        self._name = name
        self._id = (
            uuid.UUID(int=random.getrandbits(128), version=4) if _id is None else _id
        )

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    def __str__(self):
        return f"Resource(name={self.name}, id={self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.name, self.id))

    def __copy__(self):
        """Copies self and returns a new instance of Resource that shares
        the same reference to the UUID that backs the original instance."""
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, name=self.name)
        instance._id = self._id
        return instance

    def __deepcopy__(self, memo):
        """Deepcopies self and returns a new instance of Resource that has a
        new instance of the UUID with the same value as the original."""
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, name=self.name)
        instance._id = uuid.UUID(self.id)
        memo[id(self)] = instance
        return instance

    def __eq__(self, other: "Resource"):
        """Checks if the current resource is equal to the other.

        To be equivalent, the two Resource instances must have the same name and IDs,
        or one or both of them must have the 'any' ID.

        Args:
            other (`Resource`): The other resource to compare to.

        Returns:
           A boolean representing whether this Resource is of the particular
           type.
        """
        if self.name == other.name:
            if self.id == "any" or other.id == "any" or self.id == other.id:
                return True
        return False
