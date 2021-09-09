from copy import deepcopy
from typing import List

from workload.operator import Operator
from workload.task import Task


class Lattice:
    def __init__(self, operators: List[Operator]):
        self.operators = operators
        self.children_dir = {}
        for op in operators:
            self.children_dir[op.unique_id] = op.children_op_ids

    def __deepcopy__(self, memo):
        return Lattice(deepcopy(self.operators, memo))

    def __repr__(self):
        return "Lattice {}".format(self.children_dir.__repr__())

    def get_children(self, operator_id: int) -> List[int]:
        return self.children_dir[operator_id]

    def add_next(self, task: Task, new_task_queue: List, curr_time: int):
        """Adds new tasks to queue if necessary when an Task completes.

        The times are a bit weird cause this effectively gets called at the end
        of the curr_time slice/beginning of the next slice.
        We assume the tasks are made available immediately
        """
        children = self.get_children(task.operator_id)
        for c in children:
            c = self.get_op(c)
            task_id = task.unique_id
            if c.is_join:
                if not c.paused_job:
                    c.paused_job = task
                    continue
                else:
                    task_id = min(task.unique_id, c.paused_job.unique_id)
                    c.paused_job = None
            if c.relative_deadline:
                d = c.relative_deadline + curr_time
            else:
                d = None
            new_task_queue.append(
                Task(task_id, c.unique_id, curr_time, deadline=d))

    def get_op(self, operator_id: int) -> Operator:
        """Assumes that the operators are in sorted and consecutive order."""
        out = self.operators[operator_id]
        assert out.unique_id == operator_id, (
            "ERROR: lattice not ordered [{} returned instead of {}]".format(
                out.unique_id, operator_id))
        return out
