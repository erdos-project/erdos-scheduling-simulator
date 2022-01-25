from schedulers.ilp_scheduler import ILPScheduler
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler
import pickle
from pathlib import Path

from absl import app, flags
from schedulers.ilp_scheduler import verify_schedule

FLAGS = flags.FLAGS

flags.DEFINE_integer('NUM_GPUS', 2, 'Number of GPUs available.')
flags.DEFINE_integer('NUM_CPUS', 10, 'Number of CPUs available.')
flags.DEFINE_integer('task_runtime', 15, 'Estimated task runtime.')
flags.DEFINE_enum('scheduler', 'z3', ['z3', 'gurobi'],
                  'Sets which ILP scheduler to use')
flags.DEFINE_bool('opt', False,
                  'False --> feasibility only; True --> maximize slack')

flags.DEFINE_integer('dep1', 0, "first dep source")
flags.DEFINE_integer('dep2', 1, "first dep target")

mini_run = False


def do_run(scheduler: ILPScheduler,
           num_tasks: int = 5,
           expected_runtime: int = 20,
           horizon: int = 45,
           num_gpus: int = 2,
           num_cpus: int = 10,
           optimize: bool = True,
           log_dir: str = None):

    print(f"Running for {num_tasks} task over horizon of {horizon}")
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
    if not mini_run:
        dependency_matrix[FLAGS.dep1][FLAGS.dep2] = True
        needs_gpu[3] = False

    out, output_cost, sched_runtime = scheduler.schedule(
        needs_gpu,
        release_times,
        absolute_deadlines,
        expected_runtimes,
        dependency_matrix,
        pinned_tasks,
        num_tasks,
        num_gpus,
        num_cpus,
        bits=13,
        optimize=optimize,
        log_dir=log_dir,
    )

    print(sched_runtime)

    verify_schedule(out[0], out[1], needs_gpu, release_times,
                    absolute_deadlines, expected_runtimes, dependency_matrix,
                    num_gpus, num_cpus)

    print(out)
    if log_dir is not None:
        with open(log_dir + f"{'opt' if optimize else 'feas'}_result.pkl",
                  'wb') as fp:
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
    if FLAGS.scheduler == 'z3':
        scheduler = Z3Scheduler()
    elif FLAGS.scheduler == 'gurobi':
        scheduler = GurobiScheduler()
    else:
        raise ValueError('Unexpected --scheduler value {FLAGS.scheduler}')
    runtimes = []
    for i in range(1, 11, 1):
        multiplier = 5 * i
        horizon = 50 * multiplier
        num_tasks = 5 * multiplier
        log_dir = (
            FLAGS.log_dir +
            (f"/tasks={num_tasks}_horizon={horizon}_ngpu=" +
             f"{FLAGS.NUM_GPUS}_ncpu={FLAGS.NUM_CPUS}_dep={FLAGS.dep1}" +
             f"-before-{FLAGS.dep2}/") if FLAGS.log_dir else None)

        if log_dir is not None and not Path(log_dir).is_dir():
            Path(log_dir).mkdir()

        if mini_run:
            runtime, output_cost = do_run(scheduler,
                                          5,
                                          5,
                                          8000,
                                          2,
                                          1,
                                          optimize=FLAGS.opt,
                                          log_dir=log_dir)

        else:
            runtime, output_cost = do_run(scheduler,
                                          num_tasks,
                                          FLAGS.task_runtime,
                                          horizon,
                                          FLAGS.NUM_GPUS,
                                          FLAGS.NUM_CPUS,
                                          optimize=FLAGS.opt,
                                          log_dir=log_dir)

        print(f"runtime: {runtime} ----- output_cost: {output_cost}")
        runtimes.append((num_tasks, runtime, output_cost))
        if runtime > 600 or mini_run:
            break
    print(runtimes)


if __name__ == '__main__':
    app.run(main)
