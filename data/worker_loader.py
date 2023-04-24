import json
import pathlib
from typing import Mapping, Optional, Sequence, Type

import absl  # noqa: F401
import yaml

import utils
from schedulers import BaseScheduler
from workers import Worker, WorkerPool, WorkerPools
from workload import Resource, Resources


class WorkerLoader(object):
    """Loads the Worker data from the JSON or YAML files.

    Args:
        worker_profile_path (`str`): The path to the file representing the topology of
            the WorkerPools. Must be in JSON or YAML format.
        scheduler (`Optional[Type[BaseScheduler]]`): The scheduler to use for
            scheduling the tasks across the Workers of a WorkerPool.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        worker_profile_path: str,
        scheduler: Optional[Type[BaseScheduler]] = None,
        _flags: Optional["absl.flags"] = None,
    ):
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)
        self._flags = _flags

        # Read the data from the file.
        extension = pathlib.Path(worker_profile_path).suffix.lower()
        with open(worker_profile_path, "r") as f:
            if extension == ".json":
                worker_data = json.load(f)
            elif extension == ".yaml" or extension == ".yml":
                worker_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported extension: {extension}")
        if len(worker_data) == 0:
            raise ValueError("Empty data returned.")

        # Create the sequence of WorkerPools from the data.
        if _flags:
            self._worker_pools = WorkerPools(
                WorkerLoader.__create_worker_pools(
                    worker_data,
                    scheduler,
                    _flags.log_dir,
                    _flags.log_file_name,
                    _flags.log_level,
                )
            )
        else:
            self._worker_pools = WorkerPools(
                WorkerLoader.__create_worker_pools(worker_data, scheduler)
            )
        self._logger.debug(
            f"Loaded {len(self._worker_pools)} worker pools from the "
            f"file located at: {worker_profile_path}"
        )

    @staticmethod
    def __create_worker_pools(
        worker_pools: Sequence[Mapping[str, str]],
        scheduler: Type[BaseScheduler],
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: Optional[str] = "debug",
    ) -> Sequence[WorkerPool]:
        """Creates a list of WorkerPools that use the given scheduler
        implementation from the entries in `worker_pools`.

        Args:
            worker_pools (`Sequence[Mapping[str, str]]`): The entries read from
                the data file.
            scheduler (`Type[BaseScheduler]`): The scheduler implementation
                to use for each of the WorkerPools.
            log_dir (`Optional[str]`): The path to the directory where the logs are
                to be stored.
            log_file (`Optional[str]`): The name of the file to log the
                execution trace to.
            log_level (`Optional[str]`): The level of logging to do
                (default = 'debug')

        Returns:
            A `Sequence[WorkerPool]` with the `Worker`s in each of the pool
            set up according to the file's entries.
        """
        constructed_worker_pools = []
        for worker_pool in worker_pools:
            workers = []
            for worker in worker_pool["workers"]:
                resources = {}
                for resource in worker["resources"]:
                    name, quantity = resource["name"], resource["quantity"]
                    resources[Resource(name=name)] = quantity
                resource_logger = utils.setup_logging(
                    name=f"Resources_{worker['name']}",
                    log_dir=log_dir,
                    log_file=log_file,
                    log_level=log_level,
                )
                worker_logger = utils.setup_logging(
                    name=worker["name"],
                    log_dir=log_dir,
                    log_file=log_file,
                    log_level=log_level,
                )
                workers.append(
                    Worker(
                        name=worker["name"],
                        resources=Resources(
                            resource_vector=resources, _logger=resource_logger
                        ),
                        _logger=worker_logger,
                    )
                )
            worker_pool_logger = utils.setup_logging(
                name=worker_pool["name"],
                log_dir=log_dir,
                log_file=log_file,
                log_level=log_level,
            )
            constructed_worker_pools.append(
                WorkerPool(
                    name=worker_pool["name"],
                    workers=workers,
                    scheduler=scheduler,
                    _logger=worker_pool_logger,
                )
            )
        return constructed_worker_pools

    def get_worker_pools(self) -> WorkerPools:
        """Retrieve the `WorkerPool`s loaded by the `WorkerLoader`.

        Returns:
            The `WorkerPools` loaded by the `WorkerLoader`.
        """
        return self._worker_pools
