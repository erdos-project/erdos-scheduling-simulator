from typing import Sequence

from workers import WorkerPool


class WorkerLoader(object):
    """Base class for worker loaders."""

    def get_worker_pools(self) -> Sequence[WorkerPool]:
        raise NotImplementedError("The `get_worker_pools()` method has not "
                                  "been implemented.")
