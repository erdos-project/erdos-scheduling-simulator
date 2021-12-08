import logging


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
    return setup_logging(name=name + '_CSV', fmt="%(message)s", date_fmt=None,
                         log_file=log_file, log_level="debug")
