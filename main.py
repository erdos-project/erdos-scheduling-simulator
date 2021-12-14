from absl import app, flags

import utils
from simulator import Simulator
from data import TaskLoader, WorkerLoader
from schedulers import EDFScheduler, LSFScheduler

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_string('log_file_name', None,
                    'Name of the file to log the results to.',
                    short_name="log")
flags.DEFINE_string('csv_file_name', None,
                    'Name of the CSV file to log the results to.',
                    short_name="csv")
flags.DEFINE_string('log_level', 'debug', 'Level of logging.')
flags.DEFINE_string('graph_path', './data/pylot-complete-graph.dot',
                    'Path of the DOT file that contains the execution graph.')
flags.DEFINE_string('profile_path', './data/pylot_profile.json',
                    'Path of the JSON profile for the Pylot execution.')
flags.DEFINE_string('resource_path', './data/pylot_resource_profile.json',
                    'Path of the Resource requirements for each Task.')
flags.DEFINE_string('worker_profile_path', './data/worker_profile.json',
                    'Path of the topology of Workers to schedule on.')

# Task related flags.
flags.DEFINE_integer('max_timestamp', None,
                     'Maximum timestamp of tasks to load from the JSON file.',
                     short_name="max")
flags.DEFINE_float('deadline_variance', 0.0,
                   'The % variance to allocate to the assigned deadline for '
                   'each task.')
flags.DEFINE_float('runtime_variance', 0.0,
                   'The % variance to allocate to the assigned runtime for '
                   'each task.')

# Scheduler related flags.
flags.DEFINE_enum('scheduler', 'edf', ['edf', 'lsf'],
                  'The scheduler to use for this execution.')
flags.DEFINE_bool('preemption', False,
                  'Enable preemption of running tasks in the scheduler.')
flags.DEFINE_float('scheduler_runtime', -1.0,
                   'The runtime to assign to each scheduler invocation.')
flags.DEFINE_float('scheduler_delay', 1.0,
                   'The delay associated with invoking a scheduler after the '
                   'release of a Task in the system.')


def main(args):
    """Main loop that loads the data from the given profile paths, and
    runs the Simulator on the data with the given scheduler.
    """
    logger = utils.setup_logging(
            name=__name__,
            log_file=FLAGS.log_file_name,
            log_level=FLAGS.log_level)
    logger.info("Starting the execution of the simulator loop.")
    logger.info("Graph File: %s", FLAGS.graph_path)
    logger.info("Profile File: %s", FLAGS.profile_path)
    logger.info("Resource File: %s", FLAGS.resource_path)

    # Load the data.
    max_timestamp = FLAGS.max_timestamp if FLAGS.max_timestamp is not None\
        else float('inf')
    task_loader = TaskLoader(graph_path=FLAGS.graph_path,
                             profile_path=FLAGS.profile_path,
                             resource_path=FLAGS.resource_path,
                             max_timestamp=max_timestamp,
                             _flags=FLAGS)

    # Load the worker topology.
    worker_loader = WorkerLoader(worker_profile_path=FLAGS.worker_profile_path,
                                 _flags=FLAGS)

    # Instantiate the scheduler based on the given flag.
    scheduler = None
    if FLAGS.scheduler == 'edf':
        scheduler = EDFScheduler(preemptive=FLAGS.preemption,
                                 runtime=FLAGS.scheduler_runtime)
    elif FLAGS.scheduler == 'lsf':
        scheduler = LSFScheduler(preemptive=FLAGS.preemption,
                                 runtime=FLAGS.scheduler_runtime)
    else:
        raise ValueError("Unsupported scheduler implementation: {}".
                         format(FLAGS.scheduler))

    # Create and run the Simulator based on the scheduler.
    simulator = Simulator(
                    worker_pools=worker_loader.get_worker_pools(),
                    scheduler=scheduler,
                    job_graph=task_loader.get_job_graph(),
                    _flags=FLAGS)
    simulator.simulate(task_graph=task_loader.get_task_graph())


if __name__ == "__main__":
    app.run(main)
