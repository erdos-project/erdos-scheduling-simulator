"""
Solver backends for the TetriSched Python API.
"""

from __future__ import annotations

import typing

import tetrisched_py.model

__all__ = [
    "CPLEXSolver",
    "GurobiSolver",
    "SolutionType",
    "SolverBackendType",
    "SolverConfig",
    "SolverSolution",
]

class CPLEXSolver:
    def __init__(self) -> None: ...
    def exportModel(self, filename: str) -> None:
        """
        Exports the converted CPLEX model to an LP file.

        Args:
          filename (str): The filename to export the model to.
        """

    def getModel(self) -> tetrisched_py.model.SolverModel:
        """
        Returns the underlying SolverModel abstraction used by this instance of CPLEXSolver.
        """

    def solveModel(self) -> SolverSolution:
        """
        Solves the CPLEX model.

        Returns:
          SolverSolution: The characteristics of the solution.
        """

    def translateModel(self, solverConfig: SolverConfig) -> None:
        """
        Translates the underlying SolverModel to a CPLEX model instance.

        Args:
          solverConfig (SolverConfig): The configuration for the solver.
        """

class GurobiSolver:
    def __init__(self) -> None: ...
    def exportModel(self, filename: str) -> None:
        """
        Exports the converted Gurobi model to an LP file.

        Args:
          filename (str): The filename to export the model to.
        """

    def getModel(self) -> tetrisched_py.model.SolverModel:
        """
        Returns the underlying SolverModel abstraction used by this instance of GurobiSolver.
        """

    def solveModel(self) -> SolverSolution:
        """
        Solves the Gurobi model.

        Returns:
          SolverSolution: The characteristics of the solution.
        """

    def translateModel(self, solverConfig: SolverConfig) -> None:
        """
        Translates the underlying SolverModel to a Gurobi model instance.

        Args:
          solverConfig (SolverConfig): The configuration for the solver.
        """

class SolutionType:
    """
    Members:

      FEASIBLE

      OPTIMAL

      INFEASIBLE

      UNBOUNDED

      UNKNOWN

      NO_SOLUTION
    """

    FEASIBLE: typing.ClassVar[SolutionType]  # value = <SolutionType.FEASIBLE: 0>
    INFEASIBLE: typing.ClassVar[SolutionType]  # value = <SolutionType.INFEASIBLE: 2>
    NO_SOLUTION: typing.ClassVar[SolutionType]  # value = <SolutionType.NO_SOLUTION: 5>
    OPTIMAL: typing.ClassVar[SolutionType]  # value = <SolutionType.OPTIMAL: 1>
    UNBOUNDED: typing.ClassVar[SolutionType]  # value = <SolutionType.UNBOUNDED: 3>
    UNKNOWN: typing.ClassVar[SolutionType]  # value = <SolutionType.UNKNOWN: 4>
    __members__: typing.ClassVar[
        dict[str, SolutionType]
    ]  # value = {'FEASIBLE': <SolutionType.FEASIBLE: 0>, 'OPTIMAL': <SolutionType.OPTIMAL: 1>, 'INFEASIBLE': <SolutionType.INFEASIBLE: 2>, 'UNBOUNDED': <SolutionType.UNBOUNDED: 3>, 'UNKNOWN': <SolutionType.UNKNOWN: 4>, 'NO_SOLUTION': <SolutionType.NO_SOLUTION: 5>}
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

class SolverBackendType:
    """
    Members:

      CPLEX

      GUROBI

      GOOGLE_CP
    """

    CPLEX: typing.ClassVar[SolverBackendType]  # value = <SolverBackendType.CPLEX: 1>
    GOOGLE_CP: typing.ClassVar[
        SolverBackendType
    ]  # value = <SolverBackendType.GOOGLE_CP: 2>
    GUROBI: typing.ClassVar[SolverBackendType]  # value = <SolverBackendType.GUROBI: 0>
    __members__: typing.ClassVar[
        dict[str, SolverBackendType]
    ]  # value = {'CPLEX': <SolverBackendType.CPLEX: 1>, 'GUROBI': <SolverBackendType.GUROBI: 0>, 'GOOGLE_CP': <SolverBackendType.GOOGLE_CP: 2>}
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

class SolverConfig:
    def __init__(self) -> None:
        """
        Initializes an empty SolverConfig.
        """

    @property
    def newSolutionTimeMs(self) -> int | None:
        """
        The total time to be spent between finding new incumbent solutions.
        """

    @newSolutionTimeMs.setter
    def newSolutionTimeMs(self, arg0: int | None) -> None: ...
    @property
    def numThreads(self) -> int | None:
        """
        The total number of threads to allocate to the Solver.
        """

    @numThreads.setter
    def numThreads(self, arg0: int | None) -> None: ...
    @property
    def totalSolverTimeMs(self) -> int | None:
        """
        The total time to be spent on the Solver.
        """

    @totalSolverTimeMs.setter
    def totalSolverTimeMs(self, arg0: int | None) -> None: ...

class SolverSolution:
    def __str__(self) -> str: ...
    def isValid(self) -> bool:
        """
        Check if the solution was valid.
        """

    @property
    def numCachedVariables(self) -> int | None:
        """
        The number of variables that were cached before.
        """

    @property
    def numConstraints(self) -> int | None:
        """
        The number of constraints in the model.
        """

    @property
    def numDeactivatedConstraints(self) -> int | None:
        """
        The number of constraints that were generated but deactivated.
        """

    @property
    def numNonZeroCoefficients(self) -> int | None:
        """
        The number of non-zero coefficients in the model.
        """

    @property
    def numSolutions(self) -> int | None:
        """
        The number of solutions found by the solver.
        """

    @property
    def numUncachedVariables(self) -> int | None:
        """
        The number of variables that were not cached before.
        """

    @property
    def numVariables(self) -> int | None:
        """
        The number of variables in the model.
        """

    @property
    def objectiveValue(self) -> float | None:
        """
        The objective value of the solution (if available).
        """

    @property
    def objectiveValueBound(self) -> float | None:
        """
        The objective value of the bound retrieved from STRL (if available).
        """

    @property
    def solutionType(self) -> SolutionType:
        """
        The type of solution returned by the solver.
        """

    @property
    def solverTimeMicroseconds(self) -> int:
        """
        The time taken by the solver to solve the model (in microseconds).
        """
