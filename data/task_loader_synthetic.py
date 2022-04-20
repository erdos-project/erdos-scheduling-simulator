import json
import logging
import sys
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401

import utils
from data import TaskLoader, TaskLoaderJSON
from workload import Job, JobGraph, Resource, Resources, Task, TaskGraph


class TaskLoaderSynthetic(object):
    """Generates a synthetic task workload.

    Args:
        num_perception_sensors (`int`): Number of (camera, lidar) sensor pairs.
        num_traffic_light_cameras (`int`): Number of traffic light sensor cameras.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        num_perception_sensors: int = 1,
        num_traffic_light_cameras: int = 1,
        _flags: Optional["absl.flags"] = None,
    ):
        # Set up the logger.
        if _flags:
            self._logger = utils.setup_logging(
                name=self.__class__.__name__,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

        # Create the synthetic JobGraph.
        (
            self._jobs,
            self._job_graph,
            runtimes,
            deadlines,
            resources,
        ) = TaskLoaderSynthetic._TaskLoaderSynthetic__create_job_graph(
            num_perception_sensors, num_traffic_light_cameras
        )

        # Create the Tasks and the TaskGraph from the Jobs.
        task_logger = utils.setup_logging(
            name="Task", log_file=_flags.log_file_name, log_level=_flags.log_level
        )
        max_timestamp = (
            _flags.max_timestamp if _flags.max_timestamp is not None else sys.max_size
        )
        self.create_tasks(
            max_timestamp,
            _flags.timestamp_difference,
            _flags.runtime_variance,
            _flags.deadline_variance,
            runtimes,
            deadlines,
            resources,
            task_logger,
        )
        self._logger.debug(f"Created {len(self._tasks)} Tasks")
        (
            self._grouped_tasks,
            self._task_graph,
        ) = TaskLoader._TaskLoader__create_task_graph(self._tasks, self._job_graph)
        self._logger.debug("Finished creating TaskGraph from loaded tasks.")

    @staticmethod
    def __create_job_graph(
        num_perception_sensors: int,
        num_traffic_light_cameras: int,
        deadline_slack_factor: float = 1.2,
    ):
        """Creates a synthetic Pylot JobGraph.

        Args:
            num_perception_sensors (`int`): Number of cameras the pipeline has.
            num_traffic_light_cameras (`int`): Number of traffic light cameras the
                pipeline has.
            deadline_slack_factor (`float`): Factor multiplied with the task runtimes
                in order to compute task deadlines.

        Returns:
            A `JobGraph` instance depicting the relation between the different
            `Job`s.
        """
        runtimes = {}
        deadlines = {}
        resources = {}
        gnss = Job(name="gnss")
        runtimes[gnss.name] = 1000
        deadlines[gnss.name] = runtimes[gnss.name] * deadline_slack_factor
        resources[gnss.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )
        imu = Job(name="imu")
        runtimes[imu.name] = 1000
        deadlines[imu.name] = runtimes[imu.name] * deadline_slack_factor
        resources[imu.name] = Resources(resource_vector={Resource("CPU", _id="any"): 1})
        localization = Job(name="localization")
        runtimes[localization.name] = 20000
        deadlines[localization.name] = (
            runtimes[localization.name] * deadline_slack_factor
        )
        resources[localization.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )
        cameras = []
        lidars = []
        detectors = []
        trackers = []
        object_localization = []
        lane_detectors = []
        for i in range(num_perception_sensors):
            cameras.append(Job(name=f"camera_{i}", pipelined=True))
            runtimes[cameras[-1].name] = 10000
            deadlines[cameras[-1].name] = (
                runtimes[cameras[-1].name] * deadline_slack_factor
            )
            resources[cameras[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            lidars.append(Job(name=f"lidar_{i}", pipelined=True))
            runtimes[lidars[-1].name] = 8000
            deadlines[lidars[-1].name] = (
                runtimes[lidars[-1].name] * deadline_slack_factor
            )
            resources[lidars[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            detectors.append(Job(name=f"detection_{i}", pipelined=True))
            runtimes[detectors[-1].name] = 130000
            deadlines[detectors[-1].name] = (
                runtimes[detectors[-1].name] * deadline_slack_factor
            )
            resources[detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )
            trackers.append(Job(name=f"tracker_{i}", pipelined=False))
            runtimes[trackers[-1].name] = 50000
            deadlines[trackers[-1].name] = (
                runtimes[trackers[-1].name] * deadline_slack_factor
            )
            resources[trackers[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 2,
                }
            )
            object_localization.append(
                Job(name=f"obj_localization_{i}", pipelined=True)
            )
            runtimes[object_localization[-1].name] = 20000
            deadlines[object_localization[-1].name] = (
                runtimes[object_localization[-1].name] * deadline_slack_factor
            )
            resources[object_localization[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 4}
            )
            lane_detectors.append(Job(name=f"lane_detection_{i}", pipelined=True))
            runtimes[lane_detectors[-1].name] = 90000
            deadlines[lane_detectors[-1].name] = (
                runtimes[lane_detectors[-1].name] * deadline_slack_factor
            )
            resources[lane_detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )

        tl_cameras = []
        tl_detectors = []
        tl_object_localization = []
        for i in range(num_traffic_light_cameras):
            tl_cameras.append(Job(name=f"traffic_light_camera_{i}", pipelined=True))
            runtimes[tl_cameras[-1].name] = 10000
            deadlines[tl_cameras[-1].name] = (
                runtimes[tl_cameras[-1].name] * deadline_slack_factor
            )
            resources[tl_cameras[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            tl_detectors.append(Job(name=f"tl_detection_{i}", pipelined=True))
            runtimes[tl_detectors[-1].name] = 95000
            deadlines[tl_detectors[-1].name] = (
                runtimes[tl_detectors[-1].name] * deadline_slack_factor
            )
            resources[tl_detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )
            tl_object_localization.append(
                Job(name=f"tl_obj_localization_{i}", pipelined=True)
            )
            runtimes[tl_object_localization[-1].name] = 10000
            deadlines[tl_object_localization[-1].name] = (
                runtimes[tl_object_localization[-1].name] * deadline_slack_factor
            )
            resources[tl_object_localization[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )

        prediction = Job(name="prediction", pipelined=False)
        runtimes[prediction.name] = 30000
        deadlines[prediction.name] = runtimes[prediction.name] * deadline_slack_factor
        resources[prediction.name] = Resources(
            resource_vector={
                Resource("GPU", _id="any"): 1,
                Resource("CPU", _id="any"): 1,
            }
        )
        planning = Job(name="planning", pipelined=False)
        runtimes[planning.name] = 50000
        deadlines[planning.name] = runtimes[planning.name] * deadline_slack_factor
        resources[planning.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 8}
        )
        control = Job(name="control", pipelined=False)
        runtimes[control.name] = 1000
        deadlines[control.name] = runtimes[control.name] * deadline_slack_factor
        resources[control.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )

        job_graph = JobGraph()

        job_graph.add_job(gnss, [localization])
        job_graph.add_job(imu, [localization])
        job_graph.add_job(
            localization,
            [planning, control] + object_localization + tl_object_localization,
        )
        # Add camera, lidars, and their perception operators.
        for i in range(num_perception_sensors):
            job_graph.add_job(
                cameras[i], [detectors[i], lane_detectors[i], trackers[i]]
            )
            if i < num_traffic_light_cameras:
                job_graph.add_job(
                    lidars[i], [object_localization[i], tl_object_localization[i]]
                )
            else:
                job_graph.add_job(lidars[i], [object_localization[i]])
            job_graph.add_job(detectors[i], [trackers[i]])
            job_graph.add_job(trackers[i], [object_localization[i]])
            job_graph.add_job(lane_detectors[i], [planning])
            job_graph.add_job(object_localization[i], [prediction])

        # Add traffic light camera operators.
        for i in range(num_traffic_light_cameras):
            job_graph.add_job(tl_cameras[i], [tl_detectors[i]])
            job_graph.add_job(tl_detectors[i], [tl_object_localization[i]])
            job_graph.add_job(tl_object_localization[i], [prediction])

        job_graph.add_job(prediction, [planning])
        job_graph.add_job(planning, [control])
        jobs = (
            [gnss, imu, localization, prediction, planning, control]
            + cameras
            + lidars
            + detectors
            + lane_detectors
            + trackers
            + object_localization
            + tl_cameras
            + tl_detectors
            + tl_object_localization
        )
        return jobs, job_graph, runtimes, deadlines, resources

    def create_tasks(
        self,
        max_timestamp: int,
        timestamp_difference: int,
        runtime_variance: int,
        deadline_variance: int,
        runtimes: Mapping[str, int],
        deadlines: Mapping[str, int],
        resources: Mapping[str, Sequence[Resources]],
        logger: Optional[logging.Logger] = None,
    ):
        self._tasks = []
        sensor_release_time = 0
        for timestamp in range(max_timestamp + 1):
            for job in self._jobs:
                # All times are in microseconds.
                runtime = utils.fuzz_time(runtimes[job.name], runtime_variance)
                deadline = sensor_release_time + utils.fuzz_time(
                    deadlines[job.name], deadline_variance
                )
                task = Task(
                    job.name,
                    job,
                    resource_requirements=resources[job.name],
                    runtime=runtime,
                    deadline=deadline,
                    timestamp=timestamp,
                    release_time=sensor_release_time,
                    _logger=logger,
                )
                self._tasks.append(task)
            sensor_release_time += timestamp_difference

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded.

        Returns:
            The set of `Job`s loaded.
        """
        return self._jobs

    def get_job_graph(self) -> JobGraph:
        """Retrieve the constructed `JobGraph`.

        Returns:
            The `JobGraph` constructed by the TaskLoader.
        """
        return self._job_graph

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoader.

        Returns:
            The set of `Task`s loaded by the TaskLoader.
        """
        return self._tasks

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoader.

        Returns:
            The `TaskGraph` constructed by the TaskLoader.
        """
        return self._task_graph
