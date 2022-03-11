import logging
import pickle

from absl import app, flags
from pathlib import Path
from typing import Optional

import utils

from schedulers.ilp_scheduler import ILPScheduler, verify_schedule
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler
from workload import Task

FLAGS = flags.FLAGS

flags.DEFINE_string('log_file_name',
                    None,
                    'Name of the file to log the results to.',
                    short_name="log")
flags.DEFINE_string('log_level', 'debug', 'Level of logging.')
flags.DEFINE_string('ilp_log_dir', None,
                    'Directory where to log ILP scheduler info.')
flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs available.')
flags.DEFINE_integer('num_cpus', 10, 'Number of CPUs available.')
flags.DEFINE_integer('task_runtime', 15, 'Estimated task runtime.')
flags.DEFINE_enum('scheduler', 'z3', ['z3', 'gurobi'],
                  'Sets which ILP scheduler to use.')
flags.DEFINE_enum('goal', 'feasibility', ['feasibility', 'max_slack'],
                  'Sets the goal of the solver.')
flags.DEFINE_integer('source_task', 0, 'Source task.')
flags.DEFINE_integer('dependent_task', 1, 'Dependent task.')


def do_run(scheduler: ILPScheduler,
           num_tasks: int = 5,
           horizon: int = 45,
           log_dir: str = None,
           logger: Optional[logging.Logger] = None):
    logger.info(f"Scheduling {num_tasks} tasks over horizon of {horizon}")
    tasks = [
        Task(f"task_{index}", None, None, FLAGS.task_runtime, horizon, None, 2)
        for index in range(0, num_tasks)
    ]
    # True if a task requires a GPU.
    needs_gpu = [True] * num_tasks
    # True if task i must finish before task j starts
    dependency_matrix = [[False for i in range(0, num_tasks)]
                         for j in range(0, num_tasks)]
    dependency_matrix[FLAGS.source_task][FLAGS.dependent_task] = True
    needs_gpu[3] = False

    resource_requirements = [[False, True] if need_gpu else [True, False]
                             for need_gpu in needs_gpu]
    out, output_cost, sched_runtime = scheduler.schedule(
        resource_requirements, tasks, dependency_matrix,
        [FLAGS.num_cpus, FLAGS.num_gpus])

    logger.info(f"Scheduler runtime {sched_runtime}")

    verify_schedule(out[0], out[1], resource_requirements, tasks,
                    dependency_matrix, [FLAGS.num_cpus, FLAGS.num_gpus])

    # TODO: Move this logic to the scheduler.
    logger.info(f"Scheduler output {out}")
    if log_dir is not None:
        with open(log_dir + f"{FLAGS.goal}_result.pkl", 'wb') as fp:
            data = {
                'resource_requirements': resource_requirements,
                'needs_gpu': needs_gpu,
                'tasks': tasks,
                'dependency_matrix': dependency_matrix,
                'model': f"{out}",
                'n_gpu': FLAGS.num_gpus,
                'n_cpus': FLAGS.num_cpus,
                'runtime': sched_runtime,
                'output_cost': int(f"{output_cost}")
            }
            pickle.dump(data, fp)
    return sched_runtime, output_cost


def main(args):
    logger = utils.setup_logging(name=__name__,
                                 log_file=FLAGS.log_file_name,
                                 log_level=FLAGS.log_level)
    if FLAGS.scheduler == 'z3':
        scheduler = Z3Scheduler(goal=FLAGS.goal, _flags=FLAGS)
    elif FLAGS.scheduler == 'gurobi':
        scheduler = GurobiScheduler(goal=FLAGS.goal, _flags=FLAGS)
    results = []
    for multiplier in range(1, 3, 1):
        horizon = 50 * multiplier
        num_tasks = 5 * multiplier
        log_dir = (FLAGS.ilp_log_dir + (
            f"/tasks={num_tasks}_horizon={horizon}_ngpu=" +
            f"{FLAGS.num_gpus}_ncpu={FLAGS.num_cpus}_dep={FLAGS.source_task}" +
            f"-before-{FLAGS.dependent_task}/") if FLAGS.ilp_log_dir else None)
        if log_dir is not None and not Path(log_dir).is_dir():
            Path(log_dir).mkdir()

        runtime, output_cost = do_run(scheduler,
                                      num_tasks,
                                      horizon,
                                      logger=logger)

        logger.info(f"runtime: {runtime} ----- output_cost: {output_cost}")
        results.append((num_tasks, runtime, output_cost))
        if runtime > 600:
            break

    logger.info(results)


if __name__ == '__main__':
    app.run(main)
