import logging
import random
import sys
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from tabulate import tabulate

# Mapping between the requested stat values and (functions, helper messages).
STATS_FUNCTIONS = {
    "min": (np.min, "Minimum"),
    "max": (np.max, "Maximum"),
    "median": (np.median, "Median"),
    "std": (np.std, "Standard Deviation"),
    "p1": (partial(np.percentile, q=1), "Percentile (1st)"),
    "p10": (partial(np.percentile, q=10), "Percentile (10th)"),
    "p25": (partial(np.percentile, q=25), "Percentile (25th)"),
    "p50": (partial(np.percentile, q=50), "Percentile (50th)"),
    "p75": (partial(np.percentile, q=75), "Percentile (75th)"),
    "p90": (partial(np.percentile, q=90), "Percentile (90th)"),
    "p99": (partial(np.percentile, q=99), "Percentile (99th)"),
    "p99.9": (partial(np.percentile, q=99.9), "Percentile (99.9th)"),
}


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
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # Set the file to log to.
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(log_file)

    # Create the logger.
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
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


def fuzz_time(time: int, variance: Tuple[int, int]) -> int:
    """Fuzz the given `time` according to the provided `variance`.

    Args:
        time (`int`): The time to fuzz.
        variance (`Tuple[int, int]`): The (minimum, maximum) % variance to fuzz
            `time` by.

    Returns:
        The fuzzed time according to the given variance.
    """
    min_variance, max_variance = variance
    return int(
        random.uniform(
            time + (time * abs(min_variance) / 100.0),
            time + (time * abs(max_variance) / 100.0),
        )
    )


def log_statistics(
    data,
    logger: logging.Logger,
    stats: Union[str, Sequence[str]] = "all",
    offset: Optional[str] = "    ",
    showfmt: str = "grid",
):
    """Logs the requested statistics from the given data.

    Users can choose from the statistics functions defined in STATS_FUNCTIONS or
    specify 'all' to show all the statistics.

    Args:
        data (`Sequence[int]`): The data to print the statistics from.
        logger (`logging.Logger`): The logger to use for logging the stats.
        stats (`Union[str, Sequence[str]]`): The stats to be logged.
        offset (`Optional[str]`): The space offset to use for logging.
        showfmt (`Optional[str]`): The output format for the statistics.
    """
    if stats == "all":
        requested_stats = [
            "median",
            "min",
            "max",
            "std",
            "p1",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p99",
            "p99.9",
        ]
    else:
        requested_stats = [stat for stat in stats]

    if showfmt == "grid":
        results = [len(data), np.mean(data)]
        headers = ["Length", "Average"]
        for stat in requested_stats:
            method, helper = STATS_FUNCTIONS[stat]
            results.append(method(data))
            headers.append(helper)
        logger.debug("\n" + tabulate([results], headers=headers, tablefmt="grid"))
    else:
        logger.debug(f"{offset}Number of values: {len(data)}")
        logger.debug(f"{offset}Average: {np.mean(data)}")
        for stat in requested_stats:
            method, helper = STATS_FUNCTIONS[stat]
            logger.debug(f"{offset}{helper}: {method(data)}")
