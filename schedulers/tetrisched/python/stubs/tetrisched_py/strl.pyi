"""
STRL primitives for the TetriSched Python API.
"""
from __future__ import annotations
import tetrisched_py
import typing

__all__ = [
    "AllocationExpression",
    "ChooseExpression",
    "Expression",
    "ExpressionType",
    "LessThanExpression",
    "MalleableChooseExpression",
    "MaxExpression",
    "MinExpression",
    "ObjectiveExpression",
    "Placement",
    "ScaleExpression",
    "SolutionResult",
    "WindowedChooseExpression",
]

class AllocationExpression(Expression):
    def __init__(
        self,
        taskName: str,
        partitionAssignments: list[tuple[tetrisched_py.Partition, int]],
        startTime: int,
        duration: int,
    ) -> None:
        """
        Initializes an AllocationExpression for the given task to be placed on the given partitions at the given startTime, running for the given duration.

        Args:
          taskName (str): The name of the task to be placed.
          partitionAssignments (list): The list of (Partition, quantity) pairs to be placed on.
          startTime (int): The start time of the task.
          duration (int): The duration of the task.
        """

class ChooseExpression(Expression):
    def __init__(
        self,
        taskName: str,
        partitions: tetrisched_py.Partitions,
        numRequiredMachines: int,
        startTime: int,
        duration: int,
        utility: float,
    ) -> None:
        """
        Initializes a ChooseExpression for the given task to be placed on `numRequiredMachines` from the given partition at the given startTime, running for the given duration.

        Args:
          taskName (str): The name of the task to be placed.
          partitions (Partitions): The Partitions to be placed on.
          numRequiredMachines (int): The number of machines required for the task.
          startTime (int): The start time of the task.
          duration (int): The duration of the task.
          utility (TETRISCHED_ILP_TYPE): The utility of the task.
        """

class Expression:
    def __str__(self) -> str: ...
    def addChild(self, child: Expression) -> None:
        """
        Adds a child to this Expression.

        Args:
          child (Expression): The child to add to this Expression.
        """
    def exportToDot(self, fileName: str) -> None:
        """
        Exports the Expression to a dot file.

        Args:
          fileName (str): The name of the dot file to export to.
        """
    def getChildren(self) -> list[Expression]:
        """
        Returns the children of this Expression.
        """
    def getNumChildren(self) -> int:
        """
        Returns the number of children of this Expression.
        """
    def getNumParents(self) -> int:
        """
        Returns the number of parents of this Expression.
        """
    def getSolution(self) -> SolutionResult | None:
        """
        Returns the solution for this Expression.
        """
    def getType(self) -> ExpressionType:
        """
        Returns the type of this Expression.
        """
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...

class ExpressionType:
    """
    Members:

      EXPR_CHOOSE

      EXPR_OBJECTIVE

      EXPR_MIN

      EXPR_MAX

      EXPR_SCALE

      EXPR_LESSTHAN

      EXPR_ALLOCATION

      EXPR_MALLEABLE_CHOOSE

      EXPR_WINDOWED_CHOOSE
    """

    EXPR_ALLOCATION: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_ALLOCATION: 6>
    EXPR_CHOOSE: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_CHOOSE: 0>
    EXPR_LESSTHAN: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_LESSTHAN: 5>
    EXPR_MALLEABLE_CHOOSE: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_MALLEABLE_CHOOSE: 7>
    EXPR_MAX: typing.ClassVar[ExpressionType]  # value = <ExpressionType.EXPR_MAX: 3>
    EXPR_MIN: typing.ClassVar[ExpressionType]  # value = <ExpressionType.EXPR_MIN: 2>
    EXPR_OBJECTIVE: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_OBJECTIVE: 1>
    EXPR_SCALE: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_SCALE: 4>
    EXPR_WINDOWED_CHOOSE: typing.ClassVar[
        ExpressionType
    ]  # value = <ExpressionType.EXPR_WINDOWED_CHOOSE: 8>
    __members__: typing.ClassVar[
        dict[str, ExpressionType]
    ]  # value = {'EXPR_CHOOSE': <ExpressionType.EXPR_CHOOSE: 0>, 'EXPR_OBJECTIVE': <ExpressionType.EXPR_OBJECTIVE: 1>, 'EXPR_MIN': <ExpressionType.EXPR_MIN: 2>, 'EXPR_MAX': <ExpressionType.EXPR_MAX: 3>, 'EXPR_SCALE': <ExpressionType.EXPR_SCALE: 4>, 'EXPR_LESSTHAN': <ExpressionType.EXPR_LESSTHAN: 5>, 'EXPR_ALLOCATION': <ExpressionType.EXPR_ALLOCATION: 6>, 'EXPR_MALLEABLE_CHOOSE': <ExpressionType.EXPR_MALLEABLE_CHOOSE: 7>, 'EXPR_WINDOWED_CHOOSE': <ExpressionType.EXPR_WINDOWED_CHOOSE: 8>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class LessThanExpression(Expression):
    def __init__(self, name: str) -> None:
        """
        Initializes a LessThanExpression with the given name.

        Args:
          name (str): The name of the LessThanExpression.
        """

class MalleableChooseExpression(Expression):
    def __init__(
        self,
        taskName: str,
        partitions: tetrisched_py.Partitions,
        resourceTimeSlots: int,
        startTime: int,
        endTime: int,
        granularity: int,
        utility: float,
    ) -> None:
        """
        Args:
          taskName (str): The name of the task to be placed.
          partitions (Partitions): The Partitions to be placed on.
          resourceTimeSlots (int): The number of resource time slots required for the task.
          startTime (int): The start time of the task.
          endTime (int): The end time of the task.
          granularity (int): The granularity of the task.
          utility (TETRISCHED_ILP_TYPE): The utility of the task.
        """

class MaxExpression(Expression):
    def __init__(self, name: str) -> None:
        """
        Initializes a MaxExpression with the given name.

        Args:
          name (str): The name of the MaxExpression.
        """

class MinExpression(Expression):
    def __init__(self, name: str) -> None:
        """
        Initializes a MinExpression with the given name.

        Args:
          name (str): The name of the MinExpression.
        """

class ObjectiveExpression(Expression):
    def __init__(self, name: str) -> None:
        """
        Initializes an empty ObjectiveExpression with the given name.

        Args:
          name (str): The name of the ObjectiveExpression.
        """

class Placement:
    def getPartitionAllocations(self) -> dict[int, set[tuple[int, int]]]:
        """
        Returns the Partition assignments for this Placement.
        """
    def isPlaced(self) -> bool:
        """
        Returns true if the Placement was placed, false otherwise.
        """
    @property
    def name(self) -> str:
        """
        The name of the Placement.
        """
    @property
    def startTime(self) -> int | None:
        """
        The start time of the Placement (if placed).
        """

class ScaleExpression(Expression):
    def __init__(self, name: str, scalingFactor: float) -> None:
        """
        Initializes a ScaleExpression with the given name and scaling factor.

        Args:
          name (str): The name of the ScaleExpression.
          scalingFactor (TETRISCHED_ILP_TYPE): The scaling factor of the ScaleExpression.
        """

class SolutionResult:
    def __str__(self) -> str: ...
    def getPlacement(self, taskName: str) -> Placement | None:
        """
        Returns the Placement for the given task.

        Args:
          taskName (str): The name of the task to get the Placement for.
        """
    @property
    def endTime(self) -> int | None:
        """
        The end time of the expression.
        """
    @property
    def startTime(self) -> int | None:
        """
        The start time of the expression.
        """
    @property
    def utility(self) -> float | None:
        """
        The utility of the expression.
        """

class WindowedChooseExpression(Expression):
    def __init__(
        self,
        taskName: str,
        partitions: tetrisched_py.Partitions,
        numRequiredMachines: int,
        startTime: int,
        duration: int,
        endTime: int,
        granularity: int,
        utility: float,
    ) -> None:
        """
        Initializes a WindowedChooseExpression for the given task to be placed on `numRequiredMachines` from the given partition between the given startTime and endTime, and running for the given duration.

        Args:
          taskName (str): The name of the task to be placed.
          partitions (Partitions): The Partitions to be placed on.
          numRequiredMachines (int): The number of machines required for the task.
          startTime (int): The start time of the task.
          duration (int): The duration of the task.
          endTime (int): The end time of the task.
          granularity (int): The granularity of the task.
          utility (TETRISCHED_ILP_TYPE): The utility of the task.
        """
