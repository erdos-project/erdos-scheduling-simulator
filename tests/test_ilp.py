from schedulers.ilp_scheduler import ILPScheduler, verify_schedule
from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler


def __do_run(
    scheduler: ILPScheduler,
    num_tasks: int = 5,
    expected_runtime: int = 20,
    horizon: int = 45,
    num_gpus: int = 2,
    num_cpus: int = 10,
    goal: str = 'max_slack',
    log_dir: str = None,
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
    dependency_matrix[0][1] = True
    needs_gpu[3] = False

    resource_requirements = [[False, True] if need_gpu else [True, False]
                             for need_gpu in needs_gpu]

    out, output_cost, sched_runtime = scheduler.schedule(
        resource_requirements,
        release_times,
        absolute_deadlines,
        expected_runtimes,
        dependency_matrix,
        pinned_tasks,
        num_tasks,
        [num_cpus, num_gpus],
        bits=13,
        goal=goal,
        log_dir=log_dir,
    )

    verify_schedule(out[0], out[1], resource_requirements, release_times,
                    absolute_deadlines, expected_runtimes, dependency_matrix,
                    [num_cpus, num_gpus])
    #  print ('GPU/CPU Placement:' ,out[0])
    #  print ('Start Time:' ,out[1])

    return sched_runtime, output_cost, out


def test_z3_ilp_success():
    """Scenario:

    Mini-instance of z3 ilp
    """
    _, output_cost, placements = __do_run(Z3Scheduler(),
                                          5,
                                          5,
                                          8000,
                                          2,
                                          1,
                                          goal='max_slack',
                                          log_dir=None)

    assert output_cost == 39955, "Z3 ILP: Wrong cost -- {output_cost}" + \
        " instead of 39955"
    assert placements == ([2, 7, 2, 2, 7], [1, 2, 2, 0,
                                            1]), "Z3 ILP: Wrong placements"


def test_gurobi_ilp_success():
    """Scenario:

    Mini-instance of gurobi ilp
    """
    _, output_cost, placements = __do_run(GurobiScheduler(),
                                          5,
                                          5,
                                          8000,
                                          2,
                                          1,
                                          goal='max_slack',
                                          log_dir=None)

    assert output_cost == 39955, f"Gurobi ILP: Wrong cost -- {output_cost}" + \
        "instead of 39945"
    assert placements == ([2, 7, 7, 2, 2],
                          [2, 1, 2, 0,
                           1]), f"Gurobi ILP: Wrong placements -- {placements}"
