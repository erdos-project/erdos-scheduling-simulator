import json
import logging
import pathlib
import sys
from copy import deepcopy
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401
import numpy as np
import yaml

from data.workload_loader import WorkloadLoader
from utils import EventTime, setup_logging
from workload import JobGraph, Workload, WorkProfile
from workload.jobs import Job
from workload.resource import Resource
from workload.resources import Resources
from workload.strategy import ExecutionStrategies, ExecutionStrategy
from workload.tasks import Task, TaskGraph


class WorkloadLoaderClockworkBursty:
    """Loads the bursty workload from Clockwork described in section 6.2, which examines
    the serving limits of a single worker.

    Both a minor and a major workload generate tasks concurrently. The minor workload
    generates 200 tasks/second for a single instance of a ResNet50 model. The major
    workload generates 1000 tasks/second distributed evenly across another set of active
    ResNet50 model instances. The set of active models is initially empty, however 1
    model is activated at a constate rate.

    Args:
        warmup_duration: Time spent executing only the minor workload for warmup.
        model_activation_period: How long to wait before activating an additional model
            for the major workload.
        max_active_models: The maximum number of models active in the major workload.
        minor_request_rate: The rate at which requests are issued to the minor workload,
            in requests/second.
        major_request_rate: The rate at which requests are issued to the major workload,
            in requests/second.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        warmup_duration: EventTime = EventTime(300, EventTime.Unit.S),
        model_activation_period: EventTime = EventTime(1, EventTime.Unit.S),
        max_active_models: int = 3600,
        minor_request_rate: float = 200.0,
        major_request_rate: float = 1000.0,
        _flags: Optional["absl.flags"] = None,
    ) -> None:
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._resource_logger = setup_logging(
                name="Resources",
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            self._rng = np.random.default_rng(_flags.random_seed)
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            self._resource_logger = setup_logging(name="Resources")
            self._rng = np.random.default_rng()

        task_graphs = {}

        # Construct minor workload.
        self._logger.info("Generating tasks for the minor workload...")
        minor_work_profile = self.__generate_work_profile("Minor_Work_Profile")
        minor_job = WorkloadLoaderClockworkBursty.__generate_job(
            "Minor_Job", minor_work_profile
        )
        minor_workload_duration = (
            warmup_duration + model_activation_period * max_active_models
        )
        minor_release_times = self.__generate_release_times(
            minor_request_rate, minor_workload_duration
        )
        for i, release_time in enumerate(minor_release_times):
            name = f"Minor_Request_{i + 1}@{release_time.time}"
            task_graph = WorkloadLoaderClockworkBursty.__generate_task_graph(
                name, minor_job, minor_work_profile, release_time, self._logger
            )
            task_graphs[task_graph.name] = task_graph
        self._logger.info("Finished generating minor workload.")

        # Construct major workload.
        self._logger.info("Generating tasks for the major workload...")
        major_work_profiles = [
            self.__generate_work_profile(f"Major_Work_Profile_{i + 1}")
            for i in range(max_active_models)
        ]
        major_jobs = [
            WorkloadLoaderClockworkBursty.__generate_job(f"Major_Job_{i + i}", profile)
            for i, profile in enumerate(major_work_profiles)
        ]
        major_release_times = self.__generate_release_times(
            major_request_rate,
            model_activation_period * max_active_models,
            start=warmup_duration,
        )
        model_selections = self._rng.integers(
            0, 3600, len(major_release_times), dtype=np.int32
        )
        # Performance optimization: assume release times are already in microseconds.
        warmup_duration_us = warmup_duration.to(EventTime.Unit.US).time
        model_activation_period_us = model_activation_period.to(EventTime.Unit.US).time
        for i, release_time in enumerate(major_release_times):
            num_active_models = min(
                max_active_models,
                (release_time.time - warmup_duration_us) // model_activation_period_us
                + 1,
            )
            chosen_model = model_selections[i] % num_active_models
            name = f"Major_Request_{i + 1}_Model_{chosen_model}@{release_time.time}"
            task_graph = WorkloadLoaderClockworkBursty.__generate_task_graph(
                name,
                major_jobs[chosen_model],
                major_work_profiles[chosen_model],
                release_time,
                self._logger,
            )
            task_graphs[task_graph.name] = task_graph

        self._logger.info("Finished generating major workload.")

        self._workload = Workload.from_task_graphs(task_graphs, _flags)

    @property
    def workload(self) -> Workload:
        return self._workload

    def __generate_release_times(
        self,
        rate: float,
        min_total_duration: EventTime,
        start: EventTime = EventTime(0, EventTime.Unit.US),
    ) -> Sequence[EventTime]:
        inter_arrival_times_us = np.array([], dtype=np.int64)
        min_total_duration_us = min_total_duration.to(EventTime.Unit.US).time
        start_us = start.to(EventTime.Unit.US).time
        expected_period_us = 1e6 / rate
        while sum(inter_arrival_times_us) < min_total_duration_us:
            last_arrival_us = sum(inter_arrival_times_us)
            expected_additional_requests = max(
                1, round((min_total_duration_us - last_arrival_us) / expected_period_us)
            )
            new_inter_arrival_times_us = self._rng.poisson(
                expected_period_us, expected_additional_requests
            )
            inter_arrival_times_us = np.concatenate(
                [inter_arrival_times_us, new_inter_arrival_times_us]
            )
        arrival_times_us = np.cumsum(inter_arrival_times_us) + start_us
        return [EventTime(int(t), EventTime.Unit.US) for t in arrival_times_us]

    def __generate_work_profile(self, name: str) -> WorkProfile:
        return WorkProfile(
            name=name,
            loading_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="RAM", _id="any"): 102}
                        ),
                        batch_size=1,
                        runtime=EventTime(8372, EventTime.Unit.US),
                    )
                ]
            ),
            execution_strategies=ExecutionStrategies(
                strategies=[
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="GPU", _id="any"): 1}
                        ),
                        batch_size=1,
                        runtime=EventTime(3322, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="GPU", _id="any"): 1}
                        ),
                        batch_size=2,
                        runtime=EventTime(5271, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="GPU", _id="any"): 1}
                        ),
                        batch_size=4,
                        runtime=EventTime(7495, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="GPU", _id="any"): 1}
                        ),
                        batch_size=8,
                        runtime=EventTime(12439, EventTime.Unit.US),
                    ),
                    ExecutionStrategy(
                        resources=Resources(
                            resource_vector={Resource(name="GPU", _id="any"): 1}
                        ),
                        batch_size=16,
                        runtime=EventTime(21729, EventTime.Unit.US),
                    ),
                ]
            ),
        )

    @staticmethod
    def __generate_job(name: str, profile: WorkProfile) -> Job:
        return Job(name=name, profile=profile)

    @staticmethod
    def __generate_task_graph(
        name: str,
        job: Job,
        profile: WorkProfile,
        release_time: EventTime,
        logger: logging.Logger,
    ) -> TaskGraph:
        task_graph_name = f"TaskGraph_{name}"
        task = Task(
            name=f"Task_{name}",
            task_graph=task_graph_name,
            job=job,
            profile=profile,
            deadline=EventTime(100, EventTime.Unit.MS),
            timestamp=release_time.time,
            release_time=release_time,
            _logger=logger,
        )
        return TaskGraph(name=task_graph_name, tasks={task: []})
