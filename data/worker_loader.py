from workers import WorkerPools


class WorkerLoader(object):
    """Base class for worker loaders."""

    def get_worker_pools(self) -> WorkerPools:
        raise NotImplementedError(
            "The `get_worker_pools()` method has not " "been implemented."
        )
