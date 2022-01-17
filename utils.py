import random
import logging
from typing import Sequence, Optional

import numpy as np


def setup_logging(
    name: str,
    fmt: str = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s",
    date_fmt: str = "%Y-%m-%d,%H:%M:%S",
    log_file: str = None,
    log_level: str = "debug",
) -> logging.Logger:
    """Sets up the logging for the module.

    Args:
        name (`str`): The name of the logger.
        fmt (`str`): The format of the logging.
        datefmt (`str`): The format of the date to be logged.
        log_file (`str`): The path of the log file to log results to.
        log_level (`str`): The level of logging to do. (DEBUG/INFO/WARN)

    Returns:
        A `logging.Logger` instance that can be used to log the required
        information.
    """
    # Set the file to log to.
    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)

    # Create the logger.
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    # Set the logger properties.
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger


def setup_csv_logging(name: str, log_file: str) -> logging.Logger:
    """Sets up the CSV logging for the module.

    The CSV provides the data required to plot the performance characteristics
    of the scheduler.

    Args:
        name (`str`): The name of the logger.
        log_file (`str`): The path of the log file to log results to.

    Returns:
        A `logging.Logger` instance that logs the required information to the
        given CSV file.
    """
    return setup_logging(name=name + '_CSV',
                         fmt="%(message)s",
                         date_fmt=None,
                         log_file=log_file,
                         log_level="debug")


def fuzz_time(time: float,
              variance: float,
              positive: Optional[bool] = True) -> float:
    """Fuzz the given `time` according to the provided `variance`.

    Args:
        time (`float`): The time to fuzz.
        variance (`float`): The % variance to fuzz `time` by.
        positive (`Optional[bool]`): If True, the fuzzing only increases the
            time.

    Returns:
        The fuzzed time according to the given variance.
    """
    if positive:
        return random.uniform(time, time + (time * abs(variance) / 100.0))
    else:
        return max(
            0,
            random.uniform(time - (time * abs(variance) / 100.0),
                           time + (time * abs(variance) / 100.0)))


def log_statistics(data: Sequence[float],
                   logger: logging.Logger,
                   offset: Optional[str] = "    "):
    """Logs the required statistics from the given data.

    Args:
        data (`Sequence[float]`): The data to print the statistics from.
        logger (`logging.Logger`): The logger to use for logging the stats.
        offset (`Optional[str]`): The space offset to use for logging.
    """
    logger.debug("{}Number of values: {}".format(offset, len(data)))
    logger.debug("{}Average: {}".format(offset, np.mean(data)))
    logger.debug("{}Median: {}".format(offset, np.median(data)))
    logger.debug("{}Minimum: {}".format(offset, np.min(data)))
    logger.debug("{}Maximum: {}".format(offset, np.max(data)))
    logger.debug("{}Standard Deviation: {}".format(offset, np.std(data)))
    logger.debug("{}Percentile (1st): {}".format(offset,
                                                 np.percentile(data, 1)))
    logger.debug("{}Percentile (10th): {}".format(offset,
                                                  np.percentile(data, 10)))
    logger.debug("{}Percentile (25th): {}".format(offset,
                                                  np.percentile(data, 25)))
    logger.debug("{}Percentile (50th): {}".format(offset,
                                                  np.percentile(data, 50)))
    logger.debug("{}Percentile (75th): {}".format(offset,
                                                  np.percentile(data, 75)))
    logger.debug("{}Percentile (90th): {}".format(offset,
                                                  np.percentile(data, 90)))
    logger.debug("{}Percentile (99th): {}".format(offset,
                                                  np.percentile(data, 99)))
    logger.debug("{}Percentile (99.9th): {}".format(offset,
                                                    np.percentile(data, 99.9)))
