"""
Modelling primitives for the TetriSched Python API.
"""
from __future__ import annotations

import typing

__all__ = [
    "Constraint",
    "ConstraintType",
    "ObjectiveFunction",
    "ObjectiveType",
    "SolverModel",
    "Variable",
    "VariableType",
]

class Constraint:
    def __init__(self, name: str, type: ConstraintType, rhs: float) -> None:
        """
        Initializes the Constraint with the given name, type and RHS.

        Args:
          name (str): The name of this Constraint.
          type (ConstraintType): The type of this Constraint.
          rhs (int): The RHS of this Constraint.
        """
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    @typing.overload
    def addTerm(self, coefficient: float, variable: Variable) -> None:
        """
        Adds a new term to the LHS of this Constraint.

        Args:
          coefficient (int): The coefficient of the new term.
          variable (Variable): The Variable of the new term.
        """
    @typing.overload
    def addTerm(self, constant: float) -> None:
        """
        Adds a new constant to the LHS of this Constraint.

        Args:
          constant (int): The constant of the new term.
        """
    @property
    def id(self) -> int:
        """
        The ID of this Constraint.
        """
    @property
    def name(self) -> str:
        """
        The name of this Constraint.
        """

class ConstraintType:
    """
    Members:

      CONSTR_LE

      CONSTR_EQ

      CONSTR_GE
    """

    CONSTR_EQ: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONSTR_EQ: 1>
    CONSTR_GE: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONSTR_GE: 2>
    CONSTR_LE: typing.ClassVar[ConstraintType]  # value = <ConstraintType.CONSTR_LE: 0>
    __members__: typing.ClassVar[
        dict[str, ConstraintType]
    ]  # value = {'CONSTR_LE': <ConstraintType.CONSTR_LE: 0>, 'CONSTR_EQ': <ConstraintType.CONSTR_EQ: 1>, 'CONSTR_GE': <ConstraintType.CONSTR_GE: 2>}
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

class ObjectiveFunction:
    def __init__(self, type: ObjectiveType) -> None:
        """
        Initializes the ObjectiveFunction with the given type.

        Args:
          type (ObjectiveType): The type of this ObjectiveFunction.
        """
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    @typing.overload
    def addTerm(self, coefficient: float, variable: Variable) -> None:
        """
        Adds a new term to the ObjectiveFunction.

        Args:
          coefficient (int): The coefficient of the new term.
          variable (Variable): The Variable of the new term.
        """
    @typing.overload
    def addTerm(self, constant: float) -> None:
        """
        Adds a new constant to the ObjectiveFunction.

        Args:
          constant (int): The constant of the new term.
        """
    def toConstraint(self, name: str, type: ConstraintType, rhs: float) -> Constraint:
        """
        Converts this ObjectiveFunction into a Constraint.

        Args:
          name (str): The name of the Constraint to be returned.
          type (ConstraintType): The type of the Constraint to be returned.
          rhs (int): The RHS of the Constraint to be returned.
        """
    @property
    def value(self) -> float:
        """
        The value of this ObjectiveFunction.
        """

class ObjectiveType:
    """
    Members:

      OBJ_MAXIMIZE

      OBJ_MINIMIZE
    """

    OBJ_MAXIMIZE: typing.ClassVar[
        ObjectiveType
    ]  # value = <ObjectiveType.OBJ_MAXIMIZE: 0>
    OBJ_MINIMIZE: typing.ClassVar[
        ObjectiveType
    ]  # value = <ObjectiveType.OBJ_MINIMIZE: 1>
    __members__: typing.ClassVar[
        dict[str, ObjectiveType]
    ]  # value = {'OBJ_MAXIMIZE': <ObjectiveType.OBJ_MAXIMIZE: 0>, 'OBJ_MINIMIZE': <ObjectiveType.OBJ_MINIMIZE: 1>}
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

class SolverModel:
    def __str__(self) -> str: ...
    def addConstraint(self, constraint: Constraint) -> None:
        """
        Adds a new Constraint to the model.

        Args:
          constraint (Constraint): The Constraint to add to the model.
        """
    def addVariable(self, variable: Variable) -> None:
        """
        Adds a new Variable to the model.

        Args:
          variable (Variable): The Variable to add to the model.
        """
    def exportModel(self, fileName: str) -> None:
        """
        Exports the model to the given file.

        Args:
          fileName (str): The name of the file to export the model to.
        """
    def setObjectiveFunction(self, objective: ObjectiveFunction) -> None:
        """
        Sets the ObjectiveFunction of the model.

        Args:
          objective (ObjectiveFunction): The ObjectiveFunction to set.
        """
    @property
    def num_constraints(self) -> int:
        """
        The number of constraints in the model.
        """
    @property
    def num_variables(self) -> int:
        """
        The number of variables in the model.
        """
    @property
    def objective_value(self) -> float:
        """
        The value of the model's objective function.
        """

class Variable:
    @typing.overload
    def __init__(self, type: VariableType, name: str) -> None:
        """
        Initializes the Variable with the given type and name.

        Args:
          type (VariableType): The type of this Variable.
          name (str): The name of this Variable.
        """
    @typing.overload
    def __init__(self, type: VariableType, name: str, lowerBound: float) -> None:
        """
        Initializes the Variable with the given type, name and a lower bound.

        Args:
          type (VariableType): The type of this Variable.
          name (str): The name of this Variable.
          lowerBound (int): The lower bound of this Variable.
        """
    @typing.overload
    def __init__(
        self, type: VariableType, name: str, lowerBound: float, upperBound: float
    ) -> None:
        """
        Initializes the Variable with the given type, name and a lower and upper bound.

        Args:
          type (VariableType): The type of this Variable.
          name (str): The name of this Variable.
          lowerBound (int): The lower bound of this Variable.
          upperBound (int): The upper bound of this Variable.
        """
    def __str__(self) -> str: ...
    def hint(self, value: float) -> None:
        """
        Provides a hint to the solver for the initial value of this Variable.

        Args:
          value (int): The value to set for this Variable.
        """
    @property
    def id(self) -> int:
        """
        The ID of this Variable.
        """
    @property
    def name(self) -> str:
        """
        The name of this Variable.
        """
    @property
    def value(self) -> float | None:
        """
        The value of this Variable.
        """

class VariableType:
    """
    Members:

      VAR_CONTINUOUS

      VAR_INTEGER

      VAR_INDICATOR
    """

    VAR_CONTINUOUS: typing.ClassVar[
        VariableType
    ]  # value = <VariableType.VAR_CONTINUOUS: 0>
    VAR_INDICATOR: typing.ClassVar[
        VariableType
    ]  # value = <VariableType.VAR_INDICATOR: 2>
    VAR_INTEGER: typing.ClassVar[VariableType]  # value = <VariableType.VAR_INTEGER: 1>
    __members__: typing.ClassVar[
        dict[str, VariableType]
    ]  # value = {'VAR_CONTINUOUS': <VariableType.VAR_CONTINUOUS: 0>, 'VAR_INTEGER': <VariableType.VAR_INTEGER: 1>, 'VAR_INDICATOR': <VariableType.VAR_INDICATOR: 2>}
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
