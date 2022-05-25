import functools
import pickle
import sys
import time

# from math import ceil, log2
from typing import Optional

import absl  # noqa: F401
from z3 import Implies, Int, Optimize, Or, Solver, unsat

import utils
from schedulers import BaseScheduler
from workers import WorkerPools
from workload import TaskGraph, TaskState


class Z3Scheduler(BaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: int = -1,
        goal: str = "max_slack",
        enforce_deadlines: bool = True,
        lookahead: int = 0,
        _flags: Optional["absl.flags"] = None,
    ):
        """Constructs a Z3 scheduler.

        Args:
            preemptive (`bool`): If `True`, the Z3 scheduler can preempt
                the tasks that are currently running.
            runtime (`int`): The runtime to return to the simulator (in us).
                If -1, the scheduler returns the actual runtime.
            goal (`str`): Goal of the scheduler run.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                `schedule()` will return None.
            lookahead (`int`): The scheduler will try to place tasks that are within
                the scheduling lookahead (in us) using estimated task release times.
        """
        super(Z3Scheduler, self).__init__(
            preemptive, runtime, lookahead, enforce_deadlines, _flags
        )
        self._goal = goal
        self._time = None
        self._task_ids_to_task = {}
        # Mapping from task id to the var storing the task start time.
        self._task_ids_to_start_time = {}
        # Mapping from task id to the var storing the task placement.
        self._task_ids_to_placement = {}
        # Mapping from task id to the var storing the cost of the placement.
        self._task_ids_to_cost = {}
        self._task_graph = None
        self._worker_pools = None
        self._placements = []

    def _add_task_timing_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                s.add(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            s.add(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.add(
                task.deadline - start_time - task.remaining_time
                == self._task_ids_to_cost[task_id]
            )

    def _add_task_dependency_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            children = self._task_graph.get_children(task)
            for child_task in children:
                # Dependent tasks need to finish before the next one.
                if child_task.id in self._task_ids_to_start_time:
                    s.add(
                        self._task_ids_to_start_time[task_id] + task.remaining_time
                        <= self._task_ids_to_start_time[child_task.id]
                    )

    def _add_task_resource_constraints(self, s, res_type_to_index_range):
        # Add constraints whether a task is placed on GPU or CPU.
        # from num_cpus to num_cpus to num_gpus.
        for task_id, task in self._task_ids_to_task.items():
            if len(task.resource_requirements) > 1:
                self._logger.error(
                    "Scheduler doesn't support multi-resource requirements"
                )
            # for res in task.resource_requirements._resource_vector.keys():
            # TODO: Add constraints for all resources once multi-dimensional
            # requirements are supported.
            resource = next(iter(task.resource_requirements._resource_vector))
            (start_index, end_index) = res_type_to_index_range[resource.name]
            s.add(self._task_ids_to_placement[task_id] >= start_index)
            s.add(self._task_ids_to_placement[task_id] < end_index)

        for index1, (t1_id, task1) in enumerate(self._task_ids_to_task.items()):
            for index2, (t2_id, task2) in enumerate(self._task_ids_to_task.items()):
                if index2 >= index1:
                    break
                disjoint = [
                    self._task_ids_to_start_time[t1_id] + task1.remaining_time
                    <= self._task_ids_to_start_time[t2_id],
                    self._task_ids_to_start_time[t2_id] + task2.remaining_time
                    <= self._task_ids_to_start_time[t1_id],
                ]
                s.add(
                    Implies(
                        self._task_ids_to_placement[t1_id]
                        == self._task_ids_to_placement[t2_id],
                        Or(disjoint),
                    )
                )

    def schedule(self, sim_time: int, task_graph: TaskGraph, worker_pools: WorkerPools):
        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, Int(0))

        self._time = sim_time
        self._task_ids_to_task = {}
        self._task_ids_to_start_time = {}
        self._task_ids_to_placement = {}
        self._task_ids_to_cost = {}
        self._task_graph = task_graph
        self._worker_pools = worker_pools

        tasks = task_graph.get_schedulable_tasks(
            sim_time, self.lookahead, self.preemptive, worker_pools
        )

        scheduler_start_time = time.time()
        s = Optimize()

        for task in tasks:
            self._task_ids_to_task[task.id] = task
            self._task_ids_to_start_time[task.id] = Int(f"t{task.id}")
            self._task_ids_to_cost[task.id] = Int(f"c{task.id}")
            self._task_ids_to_placement[task.id] = Int(f"p{task.id}")

        (
            res_type_to_index_range,
            res_index_to_wp_id,
            _,
            _,
        ) = worker_pools.get_resource_ilp_encoding()
        self._add_task_timing_constraints(s)
        self._add_task_resource_constraints(s, res_type_to_index_range)
        self._add_task_dependency_constraints(s)

        result = s.maximize(sum_costs(self._task_ids_to_cost.values()))

        schedulable = s.check()
        scheduler_end_time = time.time()
        if self.runtime == -1:
            runtime = int((scheduler_end_time - scheduler_start_time) * 1000000)
        else:
            runtime = self.runtime

        # if self._flags.ilp_log_dir is not None:
        #     log_dir = self._flags.ilp_log_dir + f"{self._goal}.smt"
        #     with open(log_dir, "w") as outfile:
        #         outfile.write(s.sexpr())

        self._cost = sys.maxsize
        self._logger.debug(f"Solver found {schedulable} solution")
        if schedulable != unsat:
            self._cost = s.lower(result)
            self._logger.debug(f"Solver found solution with cost {self._cost}")
            self._placements = []
            for task_id, task in self._task_ids_to_task.items():
                start_time = int(str(s.model()[self._task_ids_to_start_time[task_id]]))
                placement = res_index_to_wp_id[
                    int(str(s.model()[self._task_ids_to_placement[task_id]]))
                ]
                if start_time <= sim_time + runtime * 2:
                    # We only place the tasks with a start time earlier than
                    # the estimated end time of the next scheduler run.
                    # Therefore, a task can progress before the next scheduler
                    # finishes. However, the next scheduler will assume that
                    # the task is not running while considering for placement.
                    self._placements.append((task, placement, start_time))
                else:
                    self._placements.append((task, None, None))
            self._verify_schedule(
                self._worker_pools, self._task_graph, self._placements
            )
        else:
            self._placements = [
                (task, None, None) for task in self._task_ids_to_task.values()
            ]
        # Log the scheduler run.
        self.log()
        return runtime, self._placements

    def log(self):
        if self._flags is not None and self._flags.scheduler_log_file_name is not None:
            with open(self._flags.scheduler_log_file_name, "wb") as log_file:
                logged_data = {
                    "time": self._time,
                    "tasks": self._task_ids_to_task,
                    "task_graph": self._task_graph,
                    "worker_pools": self._worker_pools,
                    "lookahead": self._lookahead,
                    "runtime": self.runtime,
                    "placements": self._placements,
                    "cost": self._cost,
                }
                pickle.dump(logged_data, log_file)
