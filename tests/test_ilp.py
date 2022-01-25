from schedulers.ilp_scheduler import ILPScheduler, verify_schedule
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler

from tests.test_tasks import __create_default_task


def __do_run(
    scheduler: ILPScheduler,
    num_tasks: int = 5,
    expected_runtime: int = 20,
    horizon: int = 45,
    num_gpus: int = 2,
    num_cpus: int = 10,
    optimize: bool = True,
    dump: bool = False,
    outpath: str = None,
    dump_nx: bool = False,
    mini_run: bool = False,
):

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
        dependency_matrix[0][1] = True
        needs_gpu[3] = False

    if outpath is not None:
        smtpath = outpath + f"{'opt' if optimize else 'feas'}.smt"
    else:
        smtpath = None
    out, output_cost, sched_runtime = scheduler.schedule(needs_gpu,
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
                                                         dump=dump,
                                                         outpath=smtpath,
                                                         dump_nx=dump_nx)

    # import IPython; IPython.embed()
    verify_schedule(out[0], out[1], needs_gpu, release_times,
                    absolute_deadlines, expected_runtimes, dependency_matrix,
                    num_gpus, num_cpus)
    #  print ('GPU/CPU Placement:' ,out[0])
    #  print ('Start Time:' ,out[1])

    return sched_runtime, output_cost, out


def test_z3_ilp_success():
    """Scenario:

    Mini-instance of z3 ilp
    """
    runtime, output_cost, placements = __do_run(Z3Scheduler(),
                                                5,
                                                5,
                                                8000,
                                                2,
                                                1,
                                                optimize=True,
                                                dump=False,
                                                outpath=None,
                                                dump_nx=False,
                                                mini_run=True)
    assert output_cost == 39945, "Gurobi ILP: Wrong cost"
    assert placements == ([2, 7, 12, 2,
                           7], [3, 3, 2, 2, 2]), "Gurobi ILP: Wrong placements"


def test_gurobi_ilp_success():
    """Scenario:

    Mini-instance of z3 ilp
    """
    runtime, output_cost, placements = __do_run(GurobiScheduler(),
                                                5,
                                                5,
                                                8000,
                                                2,
                                                1,
                                                optimize=True,
                                                dump=False,
                                                outpath=None,
                                                dump_nx=False,
                                                mini_run=True)
    assert output_cost == 39945, "Z3 ILP: Wrong cost"
    assert placements == ([2.0, 2.0, 7.0, 7.0,
                           12.0], [2.0, 3.0, 3.0, 2.0,
                                   3.0]), "Z3 ILP: Wrong placements"
