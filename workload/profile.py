import uuid

from .strategy import ExecutionStrategies


class WorkProfile(object):
    """A `WorkProfile` encapsulates the execution requirements and strategies for each
    particular `Task`. A shared `WorkProfile` enables schedulers to batch requests from
    different tasks that have the same `WorkProfile`.

    Args:
        name (`str`): A name for this profile.
        execution_strategies (`ExecutionStrategies`): The list of `ExecutionStrategies`
            that can be used to execute the task.
    """

    def __init__(self, name: str, execution_strategies: ExecutionStrategies) -> None:
        self._name = name
        self._id = uuid.uuid4()
        self._execution_strategies = execution_strategies

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def execution_strategies(self) -> ExecutionStrategies:
        return self._execution_strategies
