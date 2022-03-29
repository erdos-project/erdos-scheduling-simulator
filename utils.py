import logging
import random
from typing import Optional

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
    return setup_logging(
        name=name + "_CSV",
        fmt="%(message)s",
        date_fmt=None,
        log_file=log_file,
        log_level="debug",
    )


def fuzz_time(time: int, variance: int, positive: Optional[bool] = True) -> int:
    """Fuzz the given `time` according to the provided `variance`.

    Args:
        time (`int`): The time to fuzz.
        variance (`int`): The % variance to fuzz `time` by.
        positive (`Optional[bool]`): If True, the fuzzing only increases the
            time.

    Returns:
        The fuzzed time according to the given variance.
    """
    if positive:
        return int(random.uniform(time, time + (time * abs(variance) / 100.0)))
    else:
        return max(
            0,
            int(
                random.uniform(
                    time - time * abs(variance) / 100.0,
                    time + time * abs(variance) / 100.0,
                )
            ),
        )


def log_statistics(data, logger: logging.Logger, offset: Optional[str] = "    "):
    """Logs the required statistics from the given data.

    Args:
        data (`Sequence[int]`): The data to print the statistics from.
        logger (`logging.Logger`): The logger to use for logging the stats.
        offset (`Optional[str]`): The space offset to use for logging.
    """
    logger.debug(f"{offset}Number of values: {len(data)}")
    logger.debug(f"{offset}Average: {np.mean(data)}")
    logger.debug(f"{offset}Median: {np.median(data)}")
    logger.debug(f"{offset}Minimum: {np.min(data)}")
    logger.debug(f"{offset}Maximum: {np.max(data)}")
    logger.debug(f"{offset}Standard Deviation: {np.std(data)}")
    logger.debug(f"{offset}Percentile (1st): {np.percentile(data, 1)}")
    logger.debug(f"{offset}Percentile (10th): {np.percentile(data, 10)}")
    logger.debug(f"{offset}Percentile (25th): {np.percentile(data, 25)}")
    logger.debug(f"{offset}Percentile (50th): {np.percentile(data, 50)}")
    logger.debug(f"{offset}Percentile (75th): {np.percentile(data, 75)}")
    logger.debug(f"{offset}Percentile (90th): {np.percentile(data, 90)}")
    logger.debug(f"{offset}Percentile (99th): {np.percentile(data, 99)}")
    logger.debug(f"{offset}Percentile (99.9th): {np.percentile(data, 99.9)}")
