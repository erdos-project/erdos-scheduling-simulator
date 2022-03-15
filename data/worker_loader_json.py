import json
from typing import Optional, Sequence, Mapping, Type

import absl  # noqa: F401

import utils
from schedulers import BaseScheduler
from workers import Worker, WorkerPool
from workload import Resource, Resources


class WorkerLoaderJSON(object):
    """Loads the Worker data from the JSON files.

    Args:
        worker_profile_path (`str`): The path to the JSON file representing
            the topology of the WorkerPools.
        scheduler (`Optional[Type[BaseScheduler]]`): The scheduler to use for
            scheduling the tasks across the Workers of a WorkerPool.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(self,
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
        if _flags:
            self._worker_pools = \
                    WorkerLoaderJSON._WorkerLoaderJSON__create_worker_pools(
                        worker_data, scheduler, _flags.log_file_name,
                        _flags.log_level)
        else:
            self._worker_pools = \
                    WorkerLoaderJSON._WorkerLoaderJSON__create_worker_pools(
                        worker_data, scheduler)
        self._logger.debug(
            f"Loaded {len(self._worker_pools)} worker pools from the JSON "
            f"file located at: {worker_profile_path}")

    @staticmethod
    def __create_worker_pools(
            worker_pools: Sequence[Mapping[str, str]],
            scheduler: Type[BaseScheduler],
            log_file: Optional[str] = None,
            log_level: Optional[str] = 'debug')\
            -> Sequence[WorkerPool]:
        """Creates a list of WorkerPools that use the given scheduler
        implementation from the JSON entries in `worker_pools`.

        Args:
            worker_pools (`Sequence[Mapping[str, str]]`): The JSON entries
                read from the data file.
            scheduler (`Type[BaseScheduler]`): The scheduler implementation
                to use for each of the WorkerPools.
            log_file (`Optional[str]`): The name of the file to log the
                execution trace to.
            log_level (`Optional[str]`): The level of logging to do
                (default = 'debug')

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
                resource_logger = utils.setup_logging(
                    name=f"Resources_{worker['name']}",
                    log_file=log_file,
                    log_level=log_level)
                worker_logger = utils.setup_logging(name=worker['name'],
                                                    log_file=log_file,
                                                    log_level=log_level)
                workers.append(
                    Worker(
                        name=worker['name'],
                        resources=Resources(resource_vector=resources,
                                            _logger=resource_logger),
                        _logger=worker_logger,
                    ))
            worker_pool_logger = utils.setup_logging(name=worker_pool['name'],
                                                     log_file=log_file,
                                                     log_level=log_level)
            constructed_worker_pools.append(
                WorkerPool(
                    name=worker_pool['name'],
                    workers=workers,
                    scheduler=scheduler,
                    _logger=worker_pool_logger,
                ))
        return constructed_worker_pools

    def get_worker_pools(self) -> Sequence[WorkerPool]:
        """Retrieve the `WorkerPool`s loaded by the WorkerLoaderJSON.

        Returns:
            The `Sequence[WorkerPool]` loaded by the WorkerLoaderJSON.
        """
        return self._worker_pools
