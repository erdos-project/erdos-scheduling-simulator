from typing import List

from workers.worker import Worker


class WorkerPool:
    """Holds a list of workers which automatically log seen tasks."""
    def __init__(self, num_cpus: int, num_gpus: int):
        self.workers_gpu = [Worker(i, num_gpus=1) for i in range(num_gpus)]
        self.workers_no_gpu = [Worker(i, num_gpus=0) for i in range(num_gpus , num_gpus + num_cpus)]
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.count = num_cpus + num_gpus

    def __repr__(self):
        output = ""
        for w in self.workers():
            output += "\n{}".format(w)
        return "Worker Pool: " + output

    def add_worker(self, w: Worker):
        for v in self.workers():
            assert v.unique_id != w.unique_id, (
                "ERROR: worker already added to the pool [{}]".format(
                    w.unique_id))
        if w.num_gpus < 1:
            self.workers_no_gpu.append(w)
            self.count += 1
        else:
            self.workers_gpu.append(w)
            self.count += 1

    def add_workers(self, ws: List[Worker]):
        for w in ws:
            self.add_worker(w)

    def reset(self):
        for w in self.workers():
            w.reset()

    def history(self) -> str:
        output = ""
        for w in self.workers():
            output += "\nWorker{}: {}".format(w.unique_id, w.get_history())
        return "Worker Pool History: " + output

    def workers(self) -> List[Worker]:
        return self.workers_gpu + self.workers_no_gpu
    def get_running_tasks(self):
        return [(w.unique_id, w.current_task) for w in self.workers_gpu if w.current_task] + [(w.unique_id, w.current_task) for w in self.workers_no_gpu if w.current_task]
    def get_worker(self,i):
        assert i < self.count and i >= 0, "No such worker {i}"
        return self.workers_gpu[i] if i < self.num_gpus else self.workers_no_gpu[i-self.num_gpus]
