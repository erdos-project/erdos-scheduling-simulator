from copy import copy, deepcopy
from typing import Optional, Sequence

from utils import EventTime

from .resources import Resources


class ExecutionStrategy(object):
    """A representation of a singular execution strategy for each Task.

    An execution strategy defines how a Task would execute under a given set of
    resources. Each set of `Resources` is assigned to a given runtime, along with
    other characteristics, such as `batch_size`s that define how many requests can be
    batched together.

    Args:
        resources (`Resources`): The number and type of resources required in a
            `Worker` to execute this strategy.
        batch_size (`int`): The number of maximum requests that can be batched together
            under this strategy.
        runtime (`EventTime`): The total runtime that will be incurred on the `Worker`
            if this execution strategy is chosen.
    """

    def __init__(
        self, resources: Resources, batch_size: int, runtime: EventTime
    ) -> None:
        self._resources = resources
        self._batch_size = batch_size
        self._runtime = runtime

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def runtime(self) -> EventTime:
        return self._runtime

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __add__(self, other: "ExecutionStrategy") -> "ExecutionStrategy":
        return ExecutionStrategy(
            resources=self.resources + other.resources,
            batch_size=min(self.batch_size, other.batch_size),
            runtime=self.runtime + other.runtime,
        )

    def __copy__(self) -> "ExecutionStrategy":
        return ExecutionStrategy(
            resources=copy(self.resources),
            batch_size=self.batch_size,
            runtime=copy(self.runtime),
        )

    def __deepcopy__(self, memo) -> "ExecutionStrategy":
        instance = copy(self)
        memo[id(self)] = instance
        return instance

    def __str__(self) -> str:
        return "ExecutionStrategy(resources={}, batch_size={}, runtime={})".format(
            self._resources, self._batch_size, self._runtime
        )

    def __repr__(self) -> str:
        return str(self)


class ExecutionStrategies(object):
    """A collection of `ExecutionStrategy` objects that represent individual strategies
    for executing a given Job / Task.

    Args:
        strategies (`Optional[Sequence[ExecutionStrategy]]`): A list of strategies that
            are available for the particular Job / Task.
    """

    def __init__(
        self, strategies: Optional[Sequence[ExecutionStrategy]] = None
    ) -> None:
        self._strategies = strategies if strategies else []

    def add_strategy(self, strategy: ExecutionStrategy) -> None:
        """Add the given strategy to the list of available ExecutionStrategies for the
        particular Job / Task.

        Args:
            strategy (`ExecutionStrategy`): The strategy to add to the available
                strategy list.
        """
        self._strategies.append(strategy)

    def get_fastest_strategy(self) -> Optional[ExecutionStrategy]:
        """Get the strategy that executes in the least amount of time.

        Returns:
            A reference to the `ExecutionStrategy` that executes in the least amount of
            time. If there are no available strategies, the function returns `None`.
        """
        if len(self) == 0:
            return None
        return min(self._strategies, key=lambda s: s.runtime)

    def get_slowest_strategy(self) -> Optional[ExecutionStrategy]:
        """Get the strategy that executes in the highest amount of time.

        Returns:
            A reference to the `ExecutionStrategy` that executes in the highest amount
            of time. If there are no available strategies, the function returns `None`.
        """
        if len(self) == 0:
            return None
        return max(self._strategies, key=lambda s: s.runtime)

    def __iter__(self) -> ExecutionStrategy:
        for strategy in self._strategies:
            yield strategy

    def __getitem__(self, index):
        return self._strategies[index]

    def __len__(self) -> int:
        return len(self._strategies)

    def __str__(self) -> str:
        return "ExecutionStrategies({})".format(
            ", ".join([str(strategy) for strategy in self._strategies])
        )

    def __copy__(self) -> "ExecutionStrategies":
        return ExecutionStrategies(
            strategies=[copy(strategy) for strategy in self._strategies]
        )

    def __deepcopy__(self, memo) -> "ExecutionStrategies":
        cls = self.__class__
        instance = cls.__new__(cls)
        cls.__init__(instance, strategies=[deepcopy(strategy) for strategy in self])
        memo[id(self)] = instance
        return instance

    def __repr__(self) -> str:
        return str(self)
