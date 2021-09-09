from workload.lattice import Lattice
from workload.task import Task


class Worker:
    """Creates a new worker.

    Args:
        unique_id: The id of the worker.
        num_gpus: The number of GPUs the worker has.
    """
    def __init__(self, unique_id: int, num_gpus: int = 1):
        self.unique_id = unique_id
        self.history = []
        # Current task executing on the worker.
        self.current_task = None
        self.num_gpus = num_gpus

    def __repr__(self):
        gpu_str = ("GPU " if self.num_gpus > 0 else "CPU ")
        return gpu_str + "Worker {} -- log: {}; curr_task: {}".format(
            self.unique_id, self.history, self.current_task)

    def do_job(self, task: Task, lattice: Lattice, time: int):
        assert self.current_task is None, (
            f"ERROR: Worker {self.unique_id} is still busy")
        assert not task.needs_gpu or self.num_gpus > 0, (
            f"ERROR: Worker {self.unique_id} doesn't have GPU but task {task.unique_id} needs it"
        )
        self.history.append(task)
        self.current_task = task
        task.start(lattice, time)

    def reset(self):
        self.__init__(self.unique_id)

    def get_history(self) -> str:
        out = ""
        for e in self.history:
            out += "\n  {}".format(e)
        return "[{}\n]".format(out)

    def gpu_guarded_do_job(self, task_queue, lattice: Lattice, time: int):
        # If there's no GPU don't allocate and keep looking.
        if self.num_gpus < 1:
            for i, task in enumerate(task_queue):
                if not task.needs_gpu:
                    self.do_job(task_queue.pop(i), lattice, time)
                    break
        else:  # if there's a GPU just take the next job
            self.do_job(task_queue.pop(0), lattice, time)

    def exact_match_do_job(self, task_queue, lattice: Lattice, time: int):
        for i, task in enumerate(task_queue):
            if task.needs_gpu == (self.num_gpus > 0):
                self.do_job(task_queue.pop(i), lattice, time)
                break
