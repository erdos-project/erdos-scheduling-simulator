import os
from collections import defaultdict
from itertools import groupby
from operator import attrgetter
from typing import List, Mapping, Optional, Sequence

import absl  # noqa: F401
import numpy as np
import tetrisched_py as tetrisched

from utils import EventTime
from workers import WorkerPools
from workload import Placements, Task, TaskGraph, Workload

from .tetrisched_scheduler import Partitions, TetriSchedScheduler


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
        self._log_graphs = _flags.log_graphs if _flags else False

        # The Graphene scheduler is a STRL-based scheduler, and it requires the
        # TetriSched framework to solve the STRL formulation.
        self._min_makespan_scheduler = tetrisched.Scheduler(
            self._time_discretization.time,
            tetrisched.backends.SolverBackendType.GUROBI,
            self._log_dir,
        )
        self._min_makespan_scheduler_configuration = tetrisched.SchedulerConfig()
        self._min_makespan_scheduler_configuration.optimize = (
            self._enable_optimization_passes
        )

        # Keep a hash set of the TaskGraph names that have been transformed by the
        # scheduler already.
        self._transformed_taskgraphs = set()

        # Configuration parameters for the Offline phase of the Graphene scheduler.
        # The plan-ahead multiplier is used to determine the plan-ahead window for each
        # of the TaskGraph in the Offline phase. The multiplier is multiplied by the
        # critical path length of the TaskGraph to determine the plan-ahead window.
        self._plan_ahead_multiplier = 2

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Create a Partitions object from the WorkerPools that are available.
        partitions = Partitions(worker_pools=worker_pools)

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

            # Create the rewards for the Placement times that allow STRL to construct
            # a minimum makespan schedule for this graph.
            critical_path_runtime = task_graph.critical_path_runtime
            dilated_critical_path_runtime = EventTime(
                int(critical_path_runtime.time * self._plan_ahead_multiplier),
                critical_path_runtime.unit,
            )
            placement_times = self._get_time_discretizations_until(
                current_time=sim_time,
                end_time=sim_time + dilated_critical_path_runtime,
            )
            placement_times_and_rewards = list(
                zip(
                    placement_times,
                    np.interp(
                        list(map(lambda x: x.time, placement_times)),
                        (
                            min(placement_times).time,
                            max(placement_times).time,
                        ),
                        (2, 1),
                    ),
                )
            )

            # Construct the STRL formulation for the minimum makespan scheduling of the
            # particular TaskGraph. This corresponds to the first (offline) phase of
            # the Graphene scheduling algorithm.
            objective_strl = tetrisched.strl.ObjectiveExpression(
                f"{task_graph_name}_min_makespan"
            )
            task_graph_strl = self.construct_task_graph_strl(
                current_time=sim_time,
                task_graph=task_graph,
                partitions=partitions,
                placement_times_and_rewards=placement_times_and_rewards,
            )
            if task_graph_strl is None:
                raise ValueError(f"Failed to construct the STRL for {task_graph_name}.")
            self._logger.debug(
                "[%s] Successfully constructed the minimum makespan "
                "STRL for TaskGraph %s.",
                sim_time.time,
                task_graph_name,
            )
            objective_strl.addChild(task_graph_strl)

            # Register the STRL expression with the scheduler and solve it.
            try:
                self._min_makespan_scheduler.registerSTRL(
                    objective_strl,
                    partitions.partitions,
                    sim_time.time,
                    self._min_makespan_scheduler_configuration,
                )
                self._min_makespan_scheduler.schedule(sim_time.time)
            except RuntimeError as e:
                strl_file_name = f"{task_graph_name}_error.dot"
                solver_model_file_name = f"{task_graph_name}_error.lp"
                self._logger.error(
                    "[%s] Received error with description: %s while invoking the "
                    "STRL-based minimum makespan scheduler for TaskGraph %s. Dumping "
                    "the model to %s and STRL expression to %s.",
                    sim_time.time,
                    e,
                    task_graph_name,
                    solver_model_file_name,
                    strl_file_name,
                )
                objective_strl.exportToDot(os.path.join(self._log_dir, strl_file_name))
                self._min_makespan_scheduler.exportLastSolverModel(
                    os.path.join(self._log_dir, solver_model_file_name)
                )
                raise e

            # TODO (Sukrit): Retrieve the order of placements from the solver, and
            # construct the new TaskGraph with the placements.
            if not self._min_makespan_scheduler.getLastSolverSolution().isValid():
                strl_file_name = f"{task_graph_name}_error.dot"
                solver_model_file_name = f"{task_graph_name}_error.lp"
                self._logger.error(
                    "[%s] The minimum makespan scheduler failed to find a solution "
                    "for the STRL expression of the TaskGraph %s. Dumping the model "
                    "to %s and the STRL expression to %s.",
                    sim_time.time,
                    task_graph_name,
                    solver_model_file_name,
                    strl_file_name,
                )
                raise ValueError(
                    f"Failed to find a minimum makespan solution "
                    f"for the TaskGraph {task_graph_name}."
                )

            # Retrieve the solution and check if we were able to find a valid solution.
            task_graph_solution = objective_strl.getSolution()
            if task_graph_solution is not None and task_graph_solution.utility > 0:
                self._logger.info(
                    "[%s] Successfully found a minimum makespan solution for "
                    "TaskGraph %s and the utility was %s.",
                    sim_time.time,
                    task_graph_name,
                    task_graph_solution.utility,
                )
                # Retrieve all the Placements for the Tasks in the TaskGraph and sort
                # them by their placement time.
                placements = []
                for task in task_graph.get_nodes():
                    task_placement = task_graph_solution.getPlacement(task.unique_name)
                    if task_placement is None:
                        self._logger.error(
                            "[%s] A utility for the TaskGraph %s was found, but "
                            "the Task %s has no recorded Placement object.",
                            sim_time.time,
                            task.task_graph,
                            task.unique_name,
                        )
                        raise ValueError(
                            f"Failed to find a placement for the Task "
                            f"{task.unique_name} in the TaskGraph {task.task_graph}."
                        )
                    placements.append(task_placement)

                # Sort the placements by their start time and divide them up into stages
                # by grouped start times. We enforce direct dependencies between all
                # the tasks in each stage. Within a stage, we introduce dependencies
                # between tasks that run sequentially.
                stages = groupby(
                    sorted(placements, key=attrgetter("startTime")),
                    key=attrgetter("startTime"),
                )
                stage_to_task_map: Mapping[int, List[tetrisched.strl.Placement]] = {}
                for index, stage in enumerate(stages):
                    stage_start, placements = stage
                    stage_to_task_map[index] = list(placements)
                    self._logger.debug(
                        "[%s] Constructed stage %s for %s with "
                        "start time %s and %s elements.",
                        sim_time.time,
                        index,
                        task_graph_name,
                        stage_start,
                        len(stage_to_task_map[index]),
                    )

                # Now that the stages have been constructed, we can construct an
                # adjacency matrix for the TaskGraph by adding a dependency between
                # a Task in stage i and all the Tasks in stage i - 1 that finish before.
                updated_tasks_adjacency_matrix: Mapping[Task, Sequence[Task]] = (
                    defaultdict(list)
                )
                for stage_index, stage in stage_to_task_map.items():
                    for placement in stage:
                        task = task_graph.get_task(placement.name, unique=True)
                        if task is None:
                            raise ValueError(
                                f"Failed to find the Task {placement.name} "
                                f"in TaskGraph {task_graph.name}."
                            )
                        for previous_placement in stage_to_task_map.get(
                            stage_index - 1, []
                        ):
                            previous_task = task_graph.get_task(
                                previous_placement.name, unique=True
                            )
                            if previous_task is None:
                                raise ValueError(
                                    f"Failed to find the Task {previous_placement.name} "
                                    f"in TaskGraph {task_graph.name}."
                                )
                            strategies = previous_task.available_execution_strategies
                            fastest_strategy = strategies.get_fastest_strategy()
                            task_runtime = fastest_strategy.runtime.time
                            if (
                                previous_placement.startTime + task_runtime
                                <= placement.startTime
                            ):
                                # This Task finishes before the current Task starts in
                                # the offline order, so we add a dependency between
                                # the two Tasks here.
                                updated_tasks_adjacency_matrix[previous_task].append(
                                    task
                                )

                # Now that the adjacency matrix has been constructed, we construct a
                # new TaskGraph with the updated adjacency matrix, and swap it with the
                # adjacency matrix in the original TaskGraph.
                if self._log_graphs:
                    # Dump the original TaskGraph to the log directory.
                    task_graph.to_dot(
                        os.path.join(self._log_dir, f"{task_graph_name}_original.dot")
                    )
                task_graph.update_edges(updated_tasks_adjacency_matrix)
                if self._log_graphs:
                    # Dump the updated TaskGraph to the log directory.
                    task_graph.to_dot(
                        os.path.join(self._log_dir, f"{task_graph_name}_updated.dot")
                    )
            else:
                self._logger.error(
                    "[%s] Failed to find a minimum makespan solution for TaskGraph %s.",
                    sim_time.time,
                    task_graph_name,
                )
                raise ValueError(
                    f"Failed to find a minimum makespan solution "
                    f"for the TaskGraph {task_graph_name}."
                )

        raise NotImplementedError(
            "Saving of the transformed TaskGraphs is not yet done."
        )

        # All the TaskGraphs have been transformed, call the TetriSched scheduler and
        # return the Placements.
        return super(GrapheneScheduler, self).schedule(sim_time, workload, worker_pools)
