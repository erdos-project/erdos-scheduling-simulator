import logging
import pickle

from absl import app, flags
from pathlib import Path
from typing import Optional

import utils

from schedulers.ilp_scheduler import ILPScheduler
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler

FLAGS = flags.FLAGS

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
           expected_runtime: int = 20,
           horizon: int = 45,
           num_gpus: int = 2,
           num_cpus: int = 10,
           goal: str = 'max_slack',
           log_dir: str = None,
           logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = utils.setup_logging(name=f"{num_tasks}_{horizon}")
    logger.info(
        f"Running scheduler for {num_tasks} task over horizon of {horizon}")
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
    dependency_matrix[FLAGS.source_task][FLAGS.dependent_task] = True
    needs_gpu[3] = False

    resource_requirements = [[True, False] if need_gpu else [False, True]
                             for need_gpu in needs_gpu]
    out, output_cost, sched_runtime = scheduler.schedule(
        resource_requirements,
        release_times,
        absolute_deadlines,
        expected_runtimes,
        dependency_matrix,
        pinned_tasks,
        num_tasks,
        [num_gpus, num_cpus],
        bits=13,
        goal=goal,
        log_dir=log_dir,
    )

    logger.info(f"Scheduler runtime {sched_runtime}")

    # verify_schedule(out[0], out[1], needs_gpu, release_times,
    #                 absolute_deadlines, expected_runtimes, dependency_matrix,
    #                 [num_gpus, num_cpus])

    logger.info(f"Scheduler output {out}")
    if log_dir is not None:
        with open(log_dir + f"{goal}_result.pkl", 'wb') as fp:
            data = {
                'needs_gpu': needs_gpu,
                'release_times': release_times,
                'absolute_deadlines': absolute_deadlines,
                'expected_runtimes': expected_runtimes,
                'dependency_matrix': dependency_matrix,
                'pinned_tasks': pinned_tasks,
                'model': f"{out}",
                'n_gpu': num_gpus,
                'n_cpus': num_cpus,
                'runtime': sched_runtime,
                'output_cost': int(f"{output_cost}")
            }
            pickle.dump(data, fp)
    return (sched_runtime), output_cost


def main(args):
    logger = utils.setup_logging(name="benchmark_ilp")
    if FLAGS.scheduler == 'z3':
        scheduler = Z3Scheduler()
    elif FLAGS.scheduler == 'gurobi':
        scheduler = GurobiScheduler()
    else:
        raise ValueError('Unexpected --scheduler value {FLAGS.scheduler}')
    results = []
    for multiplier in range(1, 3, 1):
        horizon = 50 * multiplier
        num_tasks = 5 * multiplier
        log_dir = (FLAGS.log_dir + (
            f"/tasks={num_tasks}_horizon={horizon}_ngpu=" +
            f"{FLAGS.num_gpus}_ncpu={FLAGS.num_cpus}_dep={FLAGS.source_task}" +
            f"-before-{FLAGS.dependent_task}/") if FLAGS.log_dir else None)

        if log_dir is not None and not Path(log_dir).is_dir():
            Path(log_dir).mkdir()

        runtime, output_cost = do_run(scheduler,
                                      num_tasks,
                                      FLAGS.task_runtime,
                                      horizon,
                                      FLAGS.num_gpus,
                                      FLAGS.num_cpus,
                                      goal=FLAGS.goal,
                                      log_dir=log_dir,
                                      logger=logger)

        logger.info(f"runtime: {runtime} ----- output_cost: {output_cost}")
        results.append((num_tasks, runtime, output_cost))
        if runtime > 600:
            break

    logger.info(results)


if __name__ == '__main__':
    app.run(main)
