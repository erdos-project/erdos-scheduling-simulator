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
flags.DEFINE_bool('opt', False, 'False --> feasibility only; True --> maximize slack')
flags.DEFINE_bool('dump', False, 'writes smtlib2 to outpath')
flags.DEFINE_string('outpath', "out", "path for output")
flags.DEFINE_bool('dump_nx', False, 'dumps networkx object')


def do_run(scheduler: ILPScheduler,
           num_tasks: int = 5,
           expected_runtime: int = 20,
           horizon: int = 45,
           num_gpus: int = 2,
           num_cpus: int = 10,
           optimize: bool = True, 
           dump: bool = False,
           outpath: str = None,
           dump_nx: bool = False):
    
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
    mini_run=False
    if not mini_run:
        dependency_matrix[0][1] = True
        needs_gpu[3] = False

    if outpath is not None:
        smtpath = outpath + f"{'opt' if optimize else 'feas'}.smt"
    else: smtpath = None
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
                             bits=13, 
                             optimize=optimize,
                             dump = dump,
                             outpath = smtpath,
                             dump_nx = dump_nx
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
    # for i in range(1, 2, 1):
        multiplier = 5 * i
        horizon = 50 * multiplier
        num_tasks = 5 * multiplier
        outpath = (FLAGS.outpath + f"/tasks={num_tasks}_horizon={horizon}_ngpu={FLAGS.NUM_GPUS}_ncpu={FLAGS.NUM_CPUS}/" if FLAGS.dump else None)
        if outpath is not None and not Path(outpath).is_dir():
            Path(outpath).mkdir()
        mini_run = False
        if mini_run:
            run_time = do_run(scheduler, 1, 20, 10,
                                        1, 1, optimize = FLAGS.opt,
                                        dump = FLAGS.dump, outpath = outpath, dump_nx = FLAGS.dump_nx)

        
        run_time = do_run(scheduler, num_tasks, FLAGS.task_runtime, horizon,
                                    FLAGS.NUM_GPUS, FLAGS.NUM_CPUS, optimize = FLAGS.opt,
                                    dump = FLAGS.dump, outpath = outpath, dump_nx = FLAGS.dump_nx)

        print(run_time)
        runtimes.append((num_tasks,run_time))
        if run_time > 600: 
            break
    print (runtimes)
    # 10 GPUs
    # gurobi :: [(25, 0.08587193489074707), (50, 0.35022997856140137), (75, 0.7291121482849121), (100, 1.3141570091247559), (125, 2.1521008014678955), (150, 3.3707449436187744), (175, 5.060914993286133), (200, 6.168315887451172), (225, 8.27920413017273), (250, 9.921762704849243)]
    # boolector :: [(25, 0.5251703262329102), (50, 2.3219339847564697), (75, 3.3040549755096436), (100, 4.810174942016602), (125, 8.076522827148438), (150, 9.844052791595459), (175, 14.263533115386963), (200, 18.89586591720581), (225, 34.64178729057312), (250, 55.9284348487854)]
    # z3 feasible :: [(25, 0.27488088607788086), (50, 0.9974880218505859), (75, 2.5175118446350098), (100, 28.96995210647583), (125, 31.289150953292847), (150, 34.28826403617859), (175, 38.13741493225098), (200, 46.34599494934082), (225, 59.39476490020752), (250, 60.48176574707031)]
    # z3opt :: [(25, 0.640681266784668), (50, 12.719839811325073), (75, 164.75238871574402), (100, TO: 600)]

    # 3 GPUs
    # z3 feasible :: 
    # gurobi :: [(25, 0.1027829647064209), (50, 0.39047813415527344), (75, 0.8470158576965332), (100, 1.4896490573883057), (125, 2.4311540126800537), (150, 3.8708841800689697), (175, 5.760190963745117), (200, 7.8410608768463135), (225, 10.270406723022461), (250, 12.776335954666138)]
    # 2 GPUs 
    # gurobi :: [(25, 0.08508110046386719), (50, 0.31741905212402344), (75, 0.7181341648101807), (100, 1.3489925861358643), (125, 2.321063280105591), (150, 3.47944712638855), (175, 5.12633490562439), (200, 7.158405303955078), (225, 9.563724756240845), (250, 11.937131881713867)]
    # z3opt :: [(25, 0.23419690132141113), (50, 0.9150850772857666), (75, 1.9437358379364014), (100, 3.5848450660705566), (125, 6.172568082809448), (150, 9.15726900100708), (175, 12.562286853790283), (200, 16.593253135681152), (225, 21.504096031188965), (250, 27.45561718940735)]
    # boolector :: [(25, 0.5381882190704346), (50, 3.617543935775757), (75, 31.951995134353638), (100, 86.47850918769836), (125, 183.3293433189392),(150,314.9739718437195), (175, TO: 600)]
if __name__ == '__main__':
    app.run(main)
