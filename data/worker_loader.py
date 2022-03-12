from typing import Sequence

from workers import WorkerPool


class WorkerLoader(object):

    def get_worker_pools(self) -> Sequence[WorkerPool]:
        return NotImplementedError("The `get_worker_pools()` method has not "
                                   "been implemented.")
