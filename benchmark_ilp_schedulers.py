from absl import app, flags

from schedulers.boolector_scheduler import BoolectorScheduler
from schedulers.ilp_scheduler import ILPScheduler
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler
import pickle
from pathlib import Path
import time

FLAGS = flags.FLAGS

flags.DEFINE_integer('NUM_GPUS', 2, 'Number of GPUs available.')
flags.DEFINE_integer('NUM_CPUS', 10, 'Number of CPUs available.')
flags.DEFINE_integer('task_runtime', 15, 'Estimated task runtime.')
flags.DEFINE_enum('scheduler', 'boolector', ['boolector', 'z3', 'gurobi'],
                  'Sets which ILP scheduler to use')
flags.DEFINE_bool('opt', True, 'False --> feasibility only; True --> maximize slack')
flags.DEFINE_bool('dump', False, 'writes smtlib2 to outpath')
flags.DEFINE_string('outpath', "out", "path for output")


def do_run(scheduler: ILPScheduler,
           num_tasks: int = 5,
           expected_runtime: int = 20,
           horizon: int = 45,
           num_gpus: int = 2,
           num_cpus: int = 10,
           optimize: bool = True, 
           dump: bool = False,
           outpath: str = None):
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

    if outpath is not None:
        smtpath = outpath + f"{'opt' if optimize else 'feas'}.smt"
    start = time.time() # NB: the time cost includes writing the file to disk
    out = scheduler.schedule(needs_gpu,
                             release_times,
                             absolute_deadlines,
                             expected_runtimes,
                             dependency_matrix,
                             pinned_tasks,
                             num_tasks,
                             num_gpus,
                             num_cpus,
                             bits=10, 
                             optimize=optimize,
                            #  dump = dump,
                            #  outpath = smtpath
                             )
    end = time.time()
    print(end - start)
    #  print ('GPU/CPU Placement:' ,out[0])
    #  print ('Start Time:' ,out[1])
    print(out)
    if outpath is not None:
        with open(outpath + f"{'opt' if optimize else 'feas'}_res.pkl", 'wb') as fp:
            data = {
                'needs_gpu' : needs_gpu,
                'release_times' : release_times,
                'absolute_deadlines' : absolute_deadlines,
                'expected_runtimes' : expected_runtimes,
                'dependency_matrix' : dependency_matrix,
                'pinned_tasks' : pinned_tasks,
                'model' : f"{out}",
                'n_gpu' : num_gpus,
                'n_cpus' : num_cpus,
                'runtime' : end-start,
            }
            pickle.dump(data, fp)
    return (end - start)


def main(args):
    if FLAGS.scheduler == 'boolector':
        scheduler = BoolectorScheduler()
    elif FLAGS.scheduler == 'z3':
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
        outpath = (FLAGS.outpath + f"/tasks={num_tasks}_horizon={horizon}_ngpu={FLAGS.NUM_GPUS}_ncpu={FLAGS.NUM_CPUS}/" if FLAGS.dump else None)
        if outpath is not None and not Path(outpath).is_dir():
            Path(outpath).mkdir()
        # import pdb; pdb.set_trace()
        run_time = do_run(scheduler, num_tasks, FLAGS.task_runtime, horizon,
                                    FLAGS.NUM_GPUS, FLAGS.NUM_CPUS, optimize = FLAGS.opt,
                                    dump = FLAGS.dump, outpath = outpath)

        print(run_time)


if __name__ == '__main__':
    app.run(main)
