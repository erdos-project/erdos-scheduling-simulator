try:
    from .tetrisched_scheduler import TetriSchedScheduler
except ImportError:
    raise ImportError(
        "TetriSchedScheduler not found. " "Please install the TetriSched package."
    )


class GrapheneScheduler(TetriSchedScheduler):
    pass
