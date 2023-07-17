import random
import uuid
from copy import copy, deepcopy

from .strategy import ExecutionStrategies


class WorkProfile(object):
    """A `WorkProfile` encapsulates the execution requirements and strategies for each
    particular `Task`. A shared `WorkProfile` enables schedulers to batch requests from
    different tasks that have the same `WorkProfile`.

    Args:
        name (`str`): A name for this profile.
        loading_strategies (`ExecutionStrategies`): The strategies used for loading this
            profile onto a `Worker`.
        execution_strategies (`ExecutionStrategies`): The list of `ExecutionStrategies`
            that can be used to execute the task.
    """

    def __init__(
        self,
        name: str,
        execution_strategies: ExecutionStrategies = ExecutionStrategies(),
        loading_strategies: ExecutionStrategies = ExecutionStrategies(),
    ) -> None:
        self._name = name
        self._id = uuid.UUID(int=random.getrandbits(128), version=4)
        self._loading_strategies = loading_strategies
        self._execution_strategies = execution_strategies

        # Used to ensure that each copy of this WorkProfile gets a unique name.
        # This is required because of naming conflicts in ILP solvers and for better
        # logging.
        self._next_copy_number = 0

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __copy__(self) -> "WorkProfile":
        self._next_copy_number += 1
        return WorkProfile(
            name=f"{self.name}_{self._next_copy_number}",
            execution_strategies=copy(self.execution_strategies),
            loading_strategies=copy(self.loading_strategies),
        )

    def __deepcopy__(self, memo) -> "WorkProfile":
        cls = self.__class__
        instance = cls.__new__(cls)
        self._next_copy_number += 1
        cls.__init__(
            instance,
            name=f"{self.name}_{self._next_copy_number}",
            execution_strategies=deepcopy(self.execution_strategies),
            loading_strategies=deepcopy(self.loading_strategies),
        )
        instance = copy(self)
        memo[id(self)] = instance
        return instance

    def __str__(self) -> str:
        return f"WorkProfile(name={self.name}, id={self.id})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return str(self._id)

    @property
    def loading_strategies(self) -> ExecutionStrategies:
        return self._loading_strategies

    @property
    def execution_strategies(self) -> ExecutionStrategies:
        return self._execution_strategies
