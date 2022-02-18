from schedulers.ilp_scheduler import ILPScheduler
from schedulers.ilp_utils import verify_schedule, needs_gpu_to_resource_requirements

import logging
import pickle
import utils

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class ILPBenchmarker(object):

    def make_subdir(self, log_dir, num_tasks, horizon, num_gpus, num_cpus):
        subdir_name = (f"/tasks={num_tasks}_horizon={horizon}_ngpu=" +
                       f"{num_gpus}_ncpu={num_cpus}")
        return (log_dir + (subdir_name) if log_dir else None)

    def benchmark(self,
                  scheduler: ILPScheduler,
                  num_tasks: int = 5,
                  expected_runtime: int = 20,
                  num_cpus: int = 10,
                  num_gpus: int = 2,
                  goal: str = 'max_slack',
                  log_dir: str = None,
                  logger: Optional[logging.Logger] = None
                  ):

        # Make workload
        (needs_gpu, resource_requirements, release_times, absolute_deadlines,
         expected_runtimes, dependency_matrix, pinned_tasks,
         num_tasks, num_resources, horizon) = self.make_workload(num_tasks,
                                                                 expected_runtime,
                                                                 num_cpus,
                                                                 num_gpus)
        # Make logging path
        log_dir = self.make_subdir(
            log_dir, num_tasks, horizon, num_gpus, num_cpus)
        # Check log_dir exists
        if log_dir is not None and not Path(log_dir).is_dir():
            Path(log_dir).mkdir()

        # Logging
        if logger is None:
            logger = utils.setup_logging(name=f"{num_tasks}_{horizon}")
        logger.info(
            f"Running scheduler for {num_tasks} task over horizon of {horizon}")

        # Run scheduling
        out, output_cost, runtime = scheduler.schedule(
            resource_requirements,
            release_times,
            absolute_deadlines,
            expected_runtimes,
            dependency_matrix,
            pinned_tasks,
            num_tasks,
            num_resources,
            bits=13,
            goal=goal,
            log_dir=log_dir,
        )

        # Log results
        logger.info(f"Scheduler output {out}")
        logger.info(f"Scheduler runtime {runtime}")

        # Check results are valid
        verify_schedule(out[0], out[1], resource_requirements, release_times,
                        absolute_deadlines, expected_runtimes, dependency_matrix,
                        [num_cpus, num_gpus])

        # Log in
        if log_dir is not None:
            with open(log_dir + f"{goal}_result.pkl", 'wb') as fp:
                data = {
                    'resource_requirements': resource_requirements,
                    'needs_gpu': needs_gpu,
                    'release_times': release_times,
                    'absolute_deadlines': absolute_deadlines,
                    'expected_runtimes': expected_runtimes,
                    'dependency_matrix': dependency_matrix,
                    'pinned_tasks': pinned_tasks,
                    'model': f"{out}",
                    'n_gpu': num_gpus,
                    'n_cpus': num_cpus,
                    'runtime': runtime,
                    'output_cost': int(f"{output_cost}")
                }
                pickle.dump(data, fp)

        logger.info(f"runtime: {runtime} ----- output_cost: {output_cost}")
        start_time = (out[0] if out else None)
        placement = (out[1] if out else None)
        return (runtime), start_time, placement, output_cost

    def make_workload(self,
                      num_tasks: int,
                      expected_runtime: int,
                      num_cpus: int,
                      num_gpus: int
                      ):
        raise NotImplementedError

    def visualize(self, start_time, placement, num_cpus, num_gpus, num_tasks, expected_runtime):
        """ Creates a visualization if it's possible """
        if start_time is None:
            return
        raise NotImplementedError


class SymmetricTasksBenchmark (ILPBenchmarker):
    def __init__(self, deps=1, cpu_process=3):
        self.deps = deps
        if deps == 1:
            self.source_task = 0
            self.dependent_task = 1
        if cpu_process is not None:
            self.cpu_process = cpu_process

    def set_deps(self, source_task, dependent_task):
        self.source_task, self.dependent_task = source_task, dependent_task

    def make_subdir(self, log_dir, num_tasks, horizon, num_gpus, num_cpus):
        subdir_name = (f"/tasks={num_tasks}_horizon={horizon}_ngpu=" +
                       f"{num_gpus}_ncpu={num_cpus}")
        if self.deps == 1:
            subdir_name = subdir_name + ("_dep={self.source_task}" +
                                         f"-before-{self.dependent_task}/")
        return (log_dir + (subdir_name) if log_dir else None)

    def make_workload(self,
                      num_tasks: int,
                      expected_runtime: int,
                      num_cpus: int,
                      num_gpus: int
                      ):
        horizon = 10 * num_tasks
        # True if a task requires a GPU.
        needs_gpu = [True] * num_tasks
        # Release time when task is ready to start.
        release_times = [2 for _ in range(0, num_tasks)]
        # Absolute deadline when task must finish.
        absolute_deadlines = [horizon for _ in range(0, num_tasks)]
        # Expected task runtime.
        expected_runtimes = [expected_runtime for _ in range(0, num_tasks)]
        # True if task i must finish before task j starts
        dependency_matrix = [[False for i in range(0, num_tasks)]
                             for j in range(0, num_tasks)]
        # Hardware index if a task is pinned to that resource (or already running
        # there).
        pinned_tasks = [None] * num_tasks
        dependency_matrix[self.source_task][self.dependent_task] = True
        needs_gpu[self.cpu_process] = False

        resource_requirements = needs_gpu_to_resource_requirements(needs_gpu)

        num_resources = [num_cpus, num_gpus]

        return (needs_gpu, resource_requirements, release_times, absolute_deadlines,
                expected_runtimes, dependency_matrix, pinned_tasks,
                num_tasks, num_resources, horizon)

    def visualize(self, start_time, placement, num_cpus, num_gpus, num_tasks, expected_runtime):
        """ Creates a visualization if it's possible """
        if start_time is None:
            return
        return


class OneLongTwoShortBenchmark (ILPBenchmarker):

    def __init__(self, height=1.0):
        self.height = height

    def make_workload(self,
                      num_tasks: int,
                      expected_runtime: int,
                      num_cpus: int,
                      num_gpus: int
                      ):
        horizon = 30 + 10 * (num_tasks - 1)
        needs_gpu = [True] * num_tasks
        # Release time when task is ready to start.
        release_times = [22 * (i//3) for i in range(0, num_tasks)]
        # Absolute deadline when task must finish.
        absolute_deadlines = [r + 30 for r in release_times]
        # Expected task runtime.
        expected_runtimes = [10, 10, 25] * \
            (num_tasks // 3) + [10] * (num_tasks % 3)
        # True if task i must finish before task j starts
        dependency_matrix = [[False for i in range(0, num_tasks)]
                             for j in range(0, num_tasks)]
        # Hardware index if a task is pinned to that resource (or already running
        # there).
        pinned_tasks = [None] * num_tasks

        resource_requirements = needs_gpu_to_resource_requirements(needs_gpu)

        num_resources = [num_cpus, num_gpus]

        return (needs_gpu, resource_requirements, release_times, absolute_deadlines,
                expected_runtimes, dependency_matrix, pinned_tasks,
                num_tasks, num_resources, horizon)

    def visualize(self, start_time, placement, num_cpus, num_gpus, num_tasks, expected_runtime):
        """ Creates a visualization if it's possible """
        if start_time is None:
            return
        arrangement = "zero edges graph"
        height = self.height
        plt.figure(figsize=(15, 8))
        for i in range(len(start_time)):
            task_runtime = 25 if (i+1) % 3 == 0 else 10
            if placement[i] == 10:
                plt.barh(y=i * height + 0.5 * height, height=height, left=start_time[i], width=task_runtime,
                         color='#A4c08A')
            else:
                plt.barh(y=i * height + 0.5 * height, height=height, left=start_time[i], width=task_runtime,
                         color="yellow")
        plt.title("Type 2 (22, cpu) task bar graph with {} task numbe for {} gpu and {} cpu {}".format(
            num_tasks, num_gpus, num_cpus, arrangement))
        plt.savefig("Type_2_22_task_bar_graph_plot_with_{}_tasks_for_{}_gpu_and_{}_cpu_{}_cpu".format(
            num_tasks, num_gpus, num_cpus, arrangement))


class StaggeredReleaseBenchmark (ILPBenchmarker):
    def __init__(self, height=1.0):
        self.height = height

    def make_workload(self,
                      num_tasks: int,
                      expected_runtime: int,
                      num_cpus: int,
                      num_gpus: int
                      ):
        horizon = 45 + 5 * (num_tasks-1)
        needs_gpu = [True] * num_tasks
        # Release time when task is ready to start.
        release_times = [5 * k for k in range(0, num_tasks)]
        # Absolute deadline when task must finish.
        absolute_deadlines = [horizon + 5 * k for k in range(0, num_tasks)]
        # Expected task runtime.
        expected_runtimes = [expected_runtime] * num_tasks
        # True if task i must finish before task j starts
        dependency_matrix = [[False for i in range(0, num_tasks)]
                             for j in range(0, num_tasks)]
        # Hardware index if a task is pinned to that resource (or already running
        # there).
        pinned_tasks = [None] * num_tasks

        resource_requirements = needs_gpu_to_resource_requirements(needs_gpu)

        num_resources = [num_cpus, num_gpus]

        return (needs_gpu, resource_requirements, release_times, absolute_deadlines,
                expected_runtimes, dependency_matrix, pinned_tasks,
                num_tasks, num_resources, horizon)

    def visualize(self, start_time, placement, num_cpus, num_gpus, num_tasks, expected_runtime):
        """ Creates a visualization if it's possible """
        if start_time is None:
            return
        arrangement = "zero edges graph"
        plt.figure(figsize=(15, 8))
        for i in range(len(start_time)):
            plt.barh(
                y=i*self.height+0.5*self.height,
                height=self.height,
                left=start_time[i],
                width=expected_runtime,
                color='#A4C08A')
        plt.title("Type 1 task bar graph with {} task numbe for {} gpu and {} cpu {}".format(
            num_tasks, num_gpus, num_cpus, arrangement))
        plt.savefig("Type_1_task_bar_graph_plot_with_{}_tasks_for_{}_gpu_and_{}_cpu_{}_debug".format(
            num_tasks, num_gpus, num_cpus, arrangement))
