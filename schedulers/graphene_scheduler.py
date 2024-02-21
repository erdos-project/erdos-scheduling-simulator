try:
    from .tetrisched_scheduler import TetriSchedScheduler
except ImportError:
    raise ImportError(
        "TetriSchedScheduler not found. " "Please install the TetriSched package."
    )

from typing import Optional

import absl  # noqa: F401

from utils import EventTime
from workers import WorkerPools
from workload import Placements, Workload


class GrapheneScheduler(TetriSchedScheduler):
    """Implements a STRL-based formulation for the Graphene scheduler.

    The Graphene paper proposes a min-JCT scheduler that works in the following two
    stages:
        1. An Offline stage that constructs an ordering for the Tasks in a DAG assuming
            a completely empty resource-time space, and modifies the DAG structure to
            correspond to the newly created execution ordering.
        2. An Online stage that schedules the tasks of the modified DAG as they are
            released using a packing-aware scheduler.

    To achieve similar goals, we provide a faithful reimplementation of the Graphene
    scheduler using the TetriSched framework:
        1. For the offline stage, we construct a STRL formulation that minimizes the
            makespan of the DAG, and use the TetriSched framework to solve it.
        2. For the online stage, we use the TetriSched framework to schedule the tasks
            of the modified DAG as they are released and maximize packing within a
            given plan-ahead window.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime.invalid(),
        lookahead: EventTime = EventTime.zero(),
        retract_schedules: bool = False,
        goal: str = "max_goodput",
        time_discretization: EventTime = EventTime(1, EventTime.Unit.US),
        plan_ahead: EventTime = EventTime.invalid(),
        log_to_file: bool = False,
        _flags: Optional["absl.flags"] = None,
    ):
        super(GrapheneScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            # The Graphene scheduler has no notion of deadlines.
            enforce_deadlines=False,
            retract_schedules=retract_schedules,
            # The Graphene scheduler will modify TaskGraphs when they arrive, but
            # otherwise, the Workload should assume task-only scheduling.
            release_taskgraphs=False,
            goal=goal,
            time_discretization=time_discretization,
            plan_ahead=plan_ahead,
            log_to_file=log_to_file,
            _flags=_flags,
        )

        # Keep a hash set of the TaskGraph names that have been transformed by the
        # scheduler already.
        self._transformed_taskgraphs = set()

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Find the task graphs that have been added to the Workload but not yet
        # transformed by the Graphene scheduler.
        for task_graph_name, task_graph in workload.task_graphs.items():
            if task_graph_name in self._transformed_taskgraphs:
                self._logger.debug(
                    "[%s] TaskGraph %s has already been transformed.",
                    sim_time.time,
                    task_graph_name,
                )
                continue
            self._logger.debug(
                "[%s] Transforming TaskGraph %s.",
                sim_time.time,
                task_graph_name,
            )

        # All the TaskGraphs have been transformed, call the TetriSched scheduler and
        # return the Placements.
        return super(GrapheneScheduler, self).schedule(sim_time, workload, worker_pools)
