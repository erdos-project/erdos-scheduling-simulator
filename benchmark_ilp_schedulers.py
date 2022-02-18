from absl import app, flags
import utils

from schedulers.z3_scheduler import Z3Scheduler
from schedulers.gurobi_scheduler import GurobiScheduler
import matplotlib.pyplot as plt

from ilp_benchmark_utils import (
    StaggeredReleaseBenchmark, OneLongTwoShortBenchmark, SymmetricTasksBenchmark)

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs available.')
flags.DEFINE_integer('num_cpus', 10, 'Number of CPUs available.')
flags.DEFINE_integer('task_runtime', 15, 'Estimated task runtime.')

flags.DEFINE_enum('scheduler', 'z3', ['z3', 'gurobi', 'eval'],
                  'Sets which ILP scheduler to use.')
flags.DEFINE_enum('goal', 'feasibility', ['feasibility', 'max_slack'],
                  'Sets the goal of the solver.')
flags.DEFINE_integer('source_task', 0, 'Source task.')
flags.DEFINE_integer('dependent_task', 1, 'Dependent task.')
flags.DEFINE_integer('cpu_process', 3, 'Task that can go on CPU.')
flags.DEFINE_enum('workload', 'SymmetricTasks',
                  ['SymmetricTasks', 'OneLongTwoShort', 'StaggeredRelease'],
                  'Type of task')
flags.DEFINE_float('bar_height', 1.0, 'Height of the bar in plot')


def main(args):
    logger = utils.setup_logging(name="benchmark_ilp")
    if FLAGS.scheduler == 'gurobi':
        schedulers = {"gurobi": GurobiScheduler()}
    elif FLAGS.scheduler == 'z3':
        schedulers = {"z3": Z3Scheduler()}
    elif FLAGS.scheduler == 'eval':
        schedulers = {"gurobi": GurobiScheduler(), "z3": Z3Scheduler()}
    else:
        raise ValueError('Unexpected --scheduler value {FLAGS.scheduler}')

    if FLAGS.workload == "SymmetricTasks":
        benchmark = SymmetricTasksBenchmark(deps=1,
                                            cpu_process=FLAGS.cpu_process)
        benchmark.set_deps(FLAGS.source_task, FLAGS.dependent_task)
    elif FLAGS.workload == "OneLongTwoShort":
        benchmark = OneLongTwoShortBenchmark(height=FLAGS.bar_height)
    elif FLAGS.workload == "StaggeredRelease":
        benchmark = StaggeredReleaseBenchmark()
    else:
        raise NotImplementedError

    results_dict = {}
    for name, scheduler in schedulers.items():
        results = []
        for num_tasks in range(5, 7, 1):

            (runtime, start_times, placement,
             output_cost) = benchmark.benchmark(scheduler,
                                                num_tasks,
                                                FLAGS.task_runtime,
                                                FLAGS.num_cpus,
                                                FLAGS.num_gpus,
                                                goal=FLAGS.goal,
                                                log_dir=FLAGS.log_dir,
                                                logger=logger)
            benchmark.visualize(start_times, placement, FLAGS.num_cpus,
                                FLAGS.num_gpus, num_tasks, FLAGS.task_runtime)
            results.append((num_tasks, runtime, output_cost))
            if runtime > 300:
                break
        results_dict[name] = results
    logger.info(results)
    plt.close('all')

    plt.title("Runtime Summary")
    for name in results_dict.keys():
        plt.plot([t for t, _, _ in results_dict[name]],
                 [r for _, r, _ in results_dict[name]],
                 label=name)
    plt.legend()
    plt.savefig("runtime_summary.png")
    plt.close('all')

    plt.title("Cost Summary")
    for name in results_dict.keys():
        plt.plot([t for t, _, _ in results_dict[name]],
                 [c for _, _, c in results_dict[name]],
                 label=name)
    plt.legend()
    plt.savefig("cost_summary.png")


if __name__ == '__main__':
    app.run(main)
