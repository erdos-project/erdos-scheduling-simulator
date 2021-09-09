from typing import List


class Task:
    """Holder of task-related data.

    Args:
        unique_id: The unique id of the task.
        operator_id: The id of the operator that released the task.
        release_time: The release time of the task.
        runtime: The runtime of the task.
        needs_gpu: True if the task needs a GPU.
    """
    def __init__(self,
                 unique_id: int,
                 operator_id: int,
                 release_time: int,
                 runtime: int = None,
                 deadline: int = None,
                 needs_gpu: bool = False):
        self.unique_id = unique_id
        self.operator_id = operator_id
        self.release_time = release_time
        self.time_remaining = -1
        self.start_time = None
        self.finish_time = None
        self.deadline = deadline
        self.needs_gpu = needs_gpu
        self.runtime = runtime

    def __repr__(self):
        return ("<Task {}; Running Op {}; Available at Time {};"
                " Executed {} to {}; Deadline: {}>".format(
                    self.unique_id, self.operator_id, self.release_time,
                    self.start_time, self.finish_time, self.deadline))

    def start(self, lattice, time: int):
        """Starts executing the task at the given time."""
        if self.runtime:
            # We know the runtime.
            self.time_remaining = self.runtime
        else:
            # Use the estimated runtime.
            self.time_remaining = lattice.get_op(
                self.operator_id).estimated_runtime
        self.start_time = time

    def step(self, step_size=1) -> int:
        """Steps the task for 1 time unit (ms)."""
        assert self.time_remaining - step_size >= 0, (
            "ERROR: already finished task; {} left".format(
                self.time_remaining))
        self.time_remaining -= step_size
        return self.time_remaining

    def finish(self, new_task_queue: List, lattice, time: int):
        """Update bookeeping upon task finish."""
        assert self.time_remaining == 0, (
            "ERROR: try to finish but not done; {} left".format(
                self.time_remaining))
        lattice.add_next(self, new_task_queue, time)
        # Cause the task is using up the current time slice.
        self.finish_time = time
        if self.deadline and self.deadline < self.finish_time:
            print("WARNING: DEADLINE MISSED [D:{}; F:{}]".format(
                self.deadline, self.finish_time))
