# Expose the abstractions of the workload from the module.
from enum import Enum


# Define the supported Branch Prediction policies.
class BranchPredictionPolicy(Enum):
    """Represents the different prediction policies used by the Schedulers."""

    WORST_CASE = 1  # Always predict the branch that will not be taken.
    BEST_CASE = 2  # Always predict the branch that will be taken.
    MAXIMUM = 3  # Always predict the branch that takes the longest.
    RANDOM = 4  # Randomly choose a branch to be taken.
    ALL = 5  # Consider all branches in the computation.


from .jobs import Job, JobGraph
from .placement import Placement, Placements
from .resource import Resource
from .resources import Resources
from .tasks import Task, TaskGraph, TaskState
from .workload import Workload
