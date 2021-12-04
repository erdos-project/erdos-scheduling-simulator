import json
from typing import Optional, Sequence, Mapping, Type

import absl

import utils
from schedulers import BaseScheduler
from workers import Worker, WorkerPool
from workload import Resource, Resources


class WorkerLoader(object):
    """Loads the Worker data from the JSON files.

    Args:
        worker_profile_path (`str`): The path to the JSON file representing
            the topology of the WorkerPools.
        scheduler (`Optional[Type[BaseScheduler]]`): The scheduler to use for
            scheduling the tasks across the Workers of a WorkerPool.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """
    def __init__(
            self,
            worker_profile_path: str,
            scheduler: Optional[Type[BaseScheduler]] = None,
            _flags: Optional['absl.flags'] = None):
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)
        self._flags = _flags

        # Read the data from the JSON path.
        with open(worker_profile_path, 'r') as f:
            worker_data = json.load(f)
        if len(worker_data) == 0:
            raise ValueError("Empty data returned.")

        # Create the sequence of WorkerPools from the JSON data.
        self._worker_pools = WorkerLoader._WorkerLoader__create_worker_pools(
                worker_data, scheduler)
        self._logger.debug(
                "Loaded {} worker pools from the JSON file located at: {}".
                format(len(self._worker_pools), worker_profile_path))

    @staticmethod
    def __create_worker_pools(
            worker_pools: Sequence[Mapping[str, str]],
            scheduler: Type[BaseScheduler],
            _flags: Optional['absl.flags'] = None) -> Sequence[WorkerPool]:
        """Creates a list of WorkerPools that use the given scheduler
        implementation from the JSON entries in `worker_pools`.

        Args:
            worker_pools (`Sequence[Mapping[str, str]]`): The JSON entries
                read from the data file.
            scheduler (`Type[BaseScheduler]`): The scheduler implementation
                to use for each of the WorkerPools.
            _flags (`absl.flags`): The flags used to initialize, if any.

        Returns:
            A `Sequence[WorkerPool]` with the `Worker`s in each of the pool
            set up according to the JSON entries.
        """
        constructed_worker_pools = []
        for worker_pool in worker_pools:
            workers = []
            for worker in worker_pool['workers']:
                resources = {}
                for resource in worker['resources']:
                    name, quantity = resource['name'], resource['quantity']
                    resources[Resource(name=name)] = quantity
                workers.append(
                    Worker(name=worker['name'],
                           resources=Resources(resource_vector=resources,
                                               _flags=_flags),
                           _flags=_flags,
                           )
                    )
            constructed_worker_pools.append(
                    WorkerPool(name=worker_pool['name'],
                               workers=workers,
                               scheduler=scheduler,
                               _flags=_flags,
                               )
                    )
        return constructed_worker_pools

    @property
    def worker_pools(self):
        return self._worker_pools
