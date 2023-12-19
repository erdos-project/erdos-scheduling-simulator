"""
Python API for TetriSched.
"""
from __future__ import annotations

import typing

from . import backends, model, strl

__all__ = ["Partition", "Partitions", "Scheduler", "backends", "model", "strl"]

class Partition:
    def __add__(self, quantity: int) -> Partition:
        """
        Adds the given quantity to this Partition.
        """
    @typing.overload
    def __init__(self, partitionId: int, partitionName: str) -> None:
        """
        Initializes an empty Partition.

        Args:
          partitionId (int): The ID of this Partition.
          partitionName (str): The name of this Partition.
        """
    @typing.overload
    def __init__(self, partitionId: int, partitionName: str, quantity: int) -> None:
        """
        Initializes a Partition with the given quantity.

        Args:
          partitionId (int): The ID of this Partition.
          partitionName (str): The name of this Partition.
          quantity (int): The quantity of this Partition.
        """
    def __len__(self) -> int:
        """
        Returns the number of Workers in this Partition.
        """
    @property
    def id(self) -> int:
        """
        The ID of this Partition.
        """
    @property
    def name(self) -> str:
        """
        The name of this Partition.
        """

class Partitions:
    def __getitem__(self, id: int) -> Partition | None:
        """
        Returns the Partition with the given ID (if exists).
        """
    def __init__(self) -> None:
        """
        Initializes an empty Partitions.
        """
    def __len__(self) -> int:
        """
        Returns the number of Partitions in this Partitions.
        """
    def addPartition(self, partition: Partition) -> None:
        """
        Adds a Partition to this Partitions.

        Args:
          partition (Partition): The Partition to add to this Partitions.
        """
    def getPartitions(self) -> list[Partition]:
        """
        Returns the Partitions in this Partitions.
        """

class Scheduler:
    def __init__(
        self,
        discretization: int,
        solverBackend: backends.SolverBackendType,
        logDir: str = "./",
        enableDynamicDiscretization: bool = False,
        maxDiscretization: int = 5,
        maxOccupancyThreshold: float = 0.8,
    ) -> None:
        """
        Initializes the Scheduler with the given backend.

        Args:
          discretization (int): The time discretization to use for the scheduler.
          solverBackend (SolverBackendType): The solver backend to use for the scheduler.
          logDir (str): The directory where the logs are to be output.
          enableDynamicDiscretization (bool): Whether to enable dynamic discretization.
          maxDiscretization (int): The maximum discretization to use for dynamic discretization.
          maxOccupancyThreshold (float): The maximum occupancy threshold to use for dynamic discretization.
        """
    def exportLastSolverModel(self, fileName: str) -> None:
        """
        Exports the model from the last invocation of the solver.

        Args:
          fileName (str): The filename to export the model to.
        """
    def getLastSolverSolution(self) -> backends.SolverSolution:
        """
        Retrieve the solution from the last invocation of the solver.
        """
    def registerSTRL(
        self,
        expression: strl.Expression,
        availablePartitions: Partitions,
        currentTime: int,
        optimize: bool = False,
        timeRangeToGranularities: list[tuple[tuple[int, int], int]] = [],
    ) -> None:
        """
        Registers the STRL expression for the scheduler to schedule from.

        Args:
          expression (Expression): The STRL expression to register.
          availablePartitions (Partitions): The available Partitions to schedule on.
          currentTime (int): The current time.
          optimize (bool): Whether to optimize the schedule.
          timeRangeToGranularities (list): The time ranges to granularities to use for dynamic discretization.
        """
    def schedule(self, currentTime: int) -> None:
        """
        Invokes the solver to schedule the registered STRL expression.

        Args:
          currentTime (int): The current time.
        """
