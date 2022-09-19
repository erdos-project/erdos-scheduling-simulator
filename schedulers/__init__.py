# Define the branch prediction policies.
from enum import Enum


class BranchPredictionPolicy(Enum):
    """Represents the different prediction policies used by the Schedulers."""

    WORST_CASE = 1  # Always predict the branch that will not be taken.
    BEST_CASE = 2  # Always predict the branch that will be taken.
    RANDOM = 3  # Randomly choose a branch to be taken.


# Expose the BaseScheduler as part of the module.
from .base_scheduler import BaseScheduler

# Scheduler implementations
from .branch_prediction_scheduler import BranchPredictionScheduler
from .edf_scheduler import EDFScheduler
from .gurobi_scheduler import GurobiScheduler, GurobiScheduler2
from .lsf_scheduler import LSFScheduler
from .z3_scheduler import Z3Scheduler
