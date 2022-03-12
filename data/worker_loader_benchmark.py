from typing import Optional, Sequence, Type

import absl

import utils
from schedulers import BaseScheduler
from workers import Worker, WorkerPool
from workload import Resource, Resources


class WorkerLoaderBenchmark(object):

    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 _flags: Optional['absl.flags'] = None):
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)
        self._flags = _flags
        if _flags:
            self._worker_pools = \
                    WorkerLoaderBenchmark._WorkerLoaderBenchmark__create_worker_pools(
                        num_cpus, num_gpus, scheduler, _flags.log_file_name,
                        _flags.log_level)
        else:
            self._worker_pools = \
                    WorkerLoaderBenchmark._WorkerLoaderBenchmark__create_worker_pools(
                        num_cpus, num_gpus, scheduler)

    @staticmethod
    def __create_worker_pools(
            num_cpus: int,
            num_gpus: int,
            scheduler: Type[BaseScheduler],
            log_file: Optional[str] = None,
            log_level: Optional[str] = 'debug') -> Sequence[WorkerPool]:
        resources = {}
        resources[Resource(name='CPU')] = num_cpus
        resources[Resource(name='GPU')] = num_gpus
        resource_logger = utils.setup_logging(name='Resources_worker',
                                              log_file=log_file,
                                              log_level=log_level)
        worker = Worker(name='Worker',
                        resources=Resources(resource_vector=resources,
                                            _logger=resource_logger),
                        _logger=utils.setup_logging(name='Worker',
                                                    log_file=log_file,
                                                    log_level=log_level))
        worker_pool = WorkerPool(name='worker_pool',
                                 workers=[worker],
                                 scheduler=scheduler,
                                 _logger=utils.setup_logging(
                                     name='worker_pool',
                                     log_file=log_file,
                                     log_level=log_level))
        return [worker_pool]

    def get_worker_pools(self) -> Sequence[WorkerPool]:
        """Retrieve the `WorkerPool`s loaded by the WorkerLoader.

        Returns:
            The `Sequence[WorkerPool]` loaded by the WorkerLoader.
        """
        return self._worker_pools
