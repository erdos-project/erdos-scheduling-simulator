import bisect
import logging
import os
import random
import sys
from enum import Enum
from functools import partial, total_ordering
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from absl import flags

try:
    from tabulate import tabulate
except ImportError:
    pass

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


@total_ordering
class EventTime:
    @total_ordering
    class Unit(Enum):
        """Represents the different units supported by the class."""

        US = 1  # Microseconds.
        MS = 1e3  # Milliseconds.
        S = 1e6  # Seconds

        def __str__(self):
            if self == EventTime.Unit.US:
                return "µs"
            elif self == EventTime.Unit.MS:
                return "ms"
            elif self == EventTime.Unit.S:
                return "s"

        def __repr__(self):
            if self == EventTime.Unit.US:
                return "EventTime.Unit.US"
            elif self == EventTime.Unit.MS:
                return "EventTime.Unit.MS"
            elif self == EventTime.Unit.S:
                return "EventTime.Unit.S"

        def __lt__(self, other):
            return self.value < other.value

        def to(self, other):
            return self.value / other.value

    __slots__ = ("_time", "_unit")
    _rng = None

    def __init__(self, time: int, unit: Unit) -> None:
        if type(unit) != EventTime.Unit:
            raise ValueError(f"The unit: {unit} is not supported.")
        if type(time) != int:
            raise ValueError(f"The type: {type(time)} is not supported for time.")

        self._time = time
        self._unit = unit
        if type(self)._rng is None:
            if hasattr(flags.FLAGS, "random_seed"):
                type(self)._rng = random.Random(flags.FLAGS.random_seed)
            else:
                type(self)._rng = random.Random(42)

    def to(self, unit: Unit) -> "EventTime":
        if unit > self.unit:
            raise ValueError(
                "Only conversions from higher granularity to lower granularity "
                "are allowed. Use `to_unchecked()`"
            )
        return EventTime(time=int(self.time * self.unit.to(unit)), unit=unit)

    def to_unchecked(self, unit: Unit) -> Tuple[float, Unit]:
        return self.time * self.unit.to(unit), unit

    def fuzz(
            self, variance: Tuple[int, int], bounds: Tuple[int, int] = (0, sys.maxsize), rng: random.Random = None
    ) -> "EventTime":
        """Fuzz the time according to the provided `variance` and within the bounds.

        Args:
            variance (`Tuple[int, int]`): The (minimum, maximum) % variance to fuzz by.
            bounds (`Tuple[int, int]`): The (minimum, maximum) bounds to fuzz within.
            rng (random.Random): The random number generator to use.  Defaults to an internal RNG if none is specified.

        Returns:
            The fuzzed time according to the given variance.
        """
        if rng is None:
            rng = type(self)._rng

        min_variance, max_variance = variance
        min_bound, max_bound = bounds
        fuzzed_time = max(
            min_bound,
            min(
                max_bound,
                rng.uniform(
                    self.time * abs(min_variance) / 100.0,
                    self.time * abs(max_variance) / 100.0,
                ),
            ),
        )

        return EventTime(round(self.time + fuzzed_time), self.unit)

    def __str__(self) -> str:
        return f"{self.time}{self.unit}"

    def __repr__(self) -> str:
        return f"EventTime(time={self.time}, unit={repr(self.unit)})"

    def __add__(self, other) -> "EventTime":
        if self.unit == other.unit:
            return EventTime(self.time + other.time, self.unit)
        elif self.unit < other.unit:
            return EventTime(self.time + other.to(self.unit).time, self.unit)
        else:
            return EventTime(self.to(other.unit).time + other.time, other.unit)

    def __sub__(self, other) -> "EventTime":
        return self + EventTime(time=-other.time, unit=other.unit)

    def __eq__(self, other) -> bool:
        return (self - other).time == 0

    def __lt__(self, other) -> bool:
        return (self - other).time < 0

    def __mul__(self, other: int) -> "EventTime":
        if type(other) != int:
            raise RuntimeError(
                f"Multiplication of EventTime with {other} of "
                f"invalid type {type(other)}."
            )
        return EventTime(time=self.time * other, unit=self.unit)

    def __hash__(self) -> int:
        return self.to(EventTime.Unit.US).time

    def __copy__(self) -> "EventTime":
        return EventTime(time=self.time, unit=self.unit)

    def is_invalid(self) -> bool:
        return self.time == -1

    @property
    def time(self) -> int:
        return self._time

    @property
    def unit(self) -> Unit:
        return self._unit

    @staticmethod
    def zero() -> "EventTime":
        return EventTime(0, EventTime.Unit.US)

    @staticmethod
    def invalid() -> "EventTime":
        return EventTime(-1, EventTime.Unit.US)


def setup_logging(
    name: str,
    fmt: Optional[str] = None,
    date_fmt: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: str = "debug",
) -> logging.Logger:
    """Sets up the logging for the module.

    Args:
        name (`str`): The name of the logger.
        fmt (`str`): The format of the logging.
        date_fmt (`str`): The format of the date to be logged.
        log_dir (`Optional[str]`): The directory where the log results are to be
            stored. If `None`, the current working directory is chosen.
        log_file (`str`): The name of the log file to log results to.
        log_level (`str`): The level of logging to do. (DEBUG/INFO/WARN)

    Returns:
        A `logging.Logger` instance that can be used to log the required
        information.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Set the logger properties.
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))

    # Set the file to log to.
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(
            os.path.join(log_dir if log_dir is not None else os.getcwd(), log_file)
        )

    # Create the logger based on the level.
    if log_level == "debug" and fmt is None:
        fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
        date_fmt = "%Y-%m-%d,%H:%M:%S"
    elif fmt is None:
        fmt = "%(name)s %(levelname)s: %(message)s"
        date_fmt = None

    # Set the formats.
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def setup_csv_logging(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Sets up the CSV logging for the module.

    The CSV provides the data required to plot the performance characteristics
    of the scheduler.

    Args:
        name (`str`): The name of the logger.
        log_dir (`Optional[str]`): The directory where the log results are to be
            stored. If `None`, the current working directory is chosen.
        log_file (`Optional[str]`): The name of the log file to store the results in.

    Returns:
        A `logging.Logger` instance that logs the required information to the
        given CSV file.
    """
    return setup_logging(
        name=name + "_CSV",
        fmt="%(message)s",
        date_fmt=None,
        log_dir=log_dir,
        log_file=log_file,
        log_level="debug",
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

    if showfmt == "grid" and "tabulate" in sys.modules:
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


def generate_monotonically_increasing_intervals(
    min_interval, max_interval, num_intervals, initial_repetitions
):
    intervals = []
    repetitions = initial_repetitions
    for i in range(num_intervals):
        next_interval = round(
            min_interval + (i / (num_intervals - 1)) * (max_interval - min_interval)
        )
        intervals.extend([next_interval] * repetitions)
        repetitions = max(1, repetitions // 2)  # Decrease repetitions by half
    return intervals


class DisjointedIntervals:
    """
    This class keeps track of a list of close intervals.
    It supports adding new intervals and checking if a new interval
    overlaps with any of the existing intervals.

    """

    def __init__(self):
        self._intervals = []

    def add(self, new_interval: tuple[int, int]) -> None:
        if new_interval[0] > new_interval[1]:
            raise ValueError(f"Invalid interval {new_interval}")
        if self.overlap(new_interval):
            raise ValueError(f"Overlap detected for {new_interval}")
        bisect.insort(self._intervals, new_interval)

    def overlap(self, new_interval: tuple[int, int]) -> bool:
        if new_interval[0] > new_interval[1]:
            raise ValueError(f"Invalid interval {new_interval}")
        if not self._intervals:
            return False

        i = bisect.bisect_left(self._intervals, new_interval)
        if i >= len(self._intervals):
            return new_interval[0] <= self._intervals[-1][1]
        elif i == 0:
            if new_interval[1] < self._intervals[i][0]:
                return False
            return True
        else:
            if (
                new_interval[0] > self._intervals[i - 1][1]
                and new_interval[1] < self._intervals[i][0]
            ):
                return False
            return True

    def placement_gap_with_left_interval(
        self,
        new_interval: tuple[int, int],
    ) -> int:
        """
        If the new interval can be added, find the gap between
        the new interval and right most existing interval
        that is to the left of this new interval.
        """
        if new_interval[0] > new_interval[1]:
            raise ValueError(f"Invalid interval {new_interval}")
        if self.overlap(new_interval):
            raise ValueError(f"Overlap detected for {new_interval}")
        if not self._intervals:
            return 0

        i = bisect.bisect_left(self._intervals, new_interval)
        if i >= len(self._intervals):
            return new_interval[0] - self._intervals[-1][1]
        elif i == 0:
            return 0
        else:
            return new_interval[0] - self._intervals[i - 1][1]
