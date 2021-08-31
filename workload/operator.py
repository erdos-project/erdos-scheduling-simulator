from typing import List
from copy import deepcopy


class Operator:
    """Holder of operator-related data.

    Args:
        unique_id: The unique id of the operator.
        children_op_ids: List of ids of the children operators.
        estimated_runtime: Estimated runtime of the operator.
        relative_deadline: The relative deadline of the operator tasks.
    """
    def __init__(self,
                 unique_id: int,
                 children_op_ids: List[int],
                 estimated_runtime: int,
                 is_join: bool = False,
                 relative_deadline: bool = False,
                 needs_gpu: bool = False,
                 name: str = None):
        # print(f"Initializing operator {unique_id}")
        self.name = name
        self.children_op_ids = children_op_ids
        self.unique_id = unique_id
        self.estimated_runtime = estimated_runtime
        self.is_join = is_join
        self.paused_job = None
        self.relative_deadline = relative_deadline
        self.needs_gpu = needs_gpu

    def __deepcopy__(self, memo):
        return Operator(self.unique_id, deepcopy(self.children_op_ids, memo),
                        self.estimated_runtime, self.is_join,
                        self.relative_deadline, self.needs_gpu, self.name)
