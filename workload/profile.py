import uuid

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
        self._id = uuid.uuid4()
        self._loading_strategies = loading_strategies
        self._execution_strategies = execution_strategies

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    def __hash__(self) -> int:
        return hash(self._id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def loading_strategies(self) -> ExecutionStrategies:
        return self._loading_strategies

    @property
    def execution_strategies(self) -> ExecutionStrategies:
        return self._execution_strategies
