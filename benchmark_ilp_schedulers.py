from absl import app, flags

from schedulers.boolector_scheduler import BoolectorScheduler
from schedulers.ilp_scheduler import ILPScheduler
from schedulers.z3_scheduler import Z3Scheduler

import time

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs available.')
flags.DEFINE_integer('num_cpus', 10, 'Number of CPUs available.')
flags.DEFINE_integer('task_runtime', 15, 'Estimated task runtime.')
flags.DEFINE_enum('scheduler', 'bolector', ['bolector', 'z3'],
                  'Sets which ILP scheduler to use')


def do_run(scheduler: ILPScheduler,
           num_tasks: int = 5,
           expected_runtime: int = 20,
           horizon: int = 45,
           num_gpus: int = 2,
           num_cpus: int = 10):
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

    dependency_matrix[0][1] = True
    needs_gpu[3] = False

    start = time.time()
    out = scheduler.schedule(needs_gpu,
                             release_times,
                             absolute_deadlines,
                             expected_runtimes,
                             dependency_matrix,
                             pinned_tasks,
                             num_tasks,
                             num_gpus,
                             num_cpus,
                             bits=10)
    end = time.time()
    print(end - start)
    #  print ('GPU/CPU Placement:' ,out[0])
    #  print ('Start Time:' ,out[1])
    print(out)
    return (end - start)


def main(args):
    if FLAGS.scheduler == 'boolector':
        scheduler = BoolectorScheduler()
    elif FLAGS.scheduler == 'z3':
        scheduler = Z3Scheduler()
    else:
        raise ValueError('Unexpected --scheduler value {FLAGS.scheduler}')
    for i in range(1, 11, 1):
        multiplier = 5 * i
        horizon = 50 * multiplier
        num_tasks = 5 * multiplier
        run_time = scheduler.do_run(num_tasks, FLAGS.task_runtime, horizon,
                                    FLAGS.NUM_GPUS, FLAGS.NUM_CPUS)
        print(run_time)


if __name__ == '__main__':
    app.run(main)
