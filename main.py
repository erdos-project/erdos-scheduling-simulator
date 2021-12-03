from absl import app, flags

import utils
from data import DataLoader

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_string('log_file_name', None,
                    'Name of the file to log the results to.')
flags.DEFINE_string('csv_file_name', None,
                    'Nmae of the CSV file to log the results to.')
flags.DEFINE_string('log_level', 'debug', 'Level of logging.')
flags.DEFINE_string('graph_path', './data/pylot-complete-graph.dot',
                    'Path of the DOT file that contains the execution graph.')
flags.DEFINE_string('profile_path', './data/pylot_profile.json',
                    'Path of the JSON profile for the Pylot execution.')
flags.DEFINE_string('resource_path', './data/pylot_resource_profile.json',
                    'Path of the Resource requirements for each Task.')
# TODO (Sukrit): Define a flag for specifying schedulers on command line.


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
    data = DataLoader(graph_path=FLAGS.graph_path,
                      profile_path=FLAGS.profile_path,
                      resource_path=FLAGS.resource_path,
                      _flags=FLAGS)

    # TODO (Sukrit): Create and run the Simulator based on the scheduler.


if __name__ == "__main__":
    app.run(main)
