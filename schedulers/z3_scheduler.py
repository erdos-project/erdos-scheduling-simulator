import functools
import pickle
import time
import sys

import absl  # noqa: F401

# from math import ceil, log2
from typing import Optional

from z3 import Int, Solver, Implies, Or, unsat, Optimize
# from z3 import If, BitVec, Bool, UGE, ULT, BitVecVal

import utils
from schedulers import BaseScheduler
from workers import WorkerPools
from workload import TaskGraph, TaskState


class Z3Scheduler(BaseScheduler):

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 goal: str = 'max_slack',
                 enforce_deadlines: bool = True,
                 scheduling_horizon: float = 0,
                 _flags: Optional['absl.flags'] = None):
        """Constructs a Z3 scheduler.

        Args:
            goal (`str`): Goal of the scheduler run.
            enforce_deadlines (`bool`): Deadlines must be met or else the
                schedule will return None.
        """
        self._preemptive = preemptive
        self._runtime = runtime
        self._goal = goal
        self._enforce_deadlines = enforce_deadlines
        self._scheduling_horizon = scheduling_horizon
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
        # Set up the logger.
        self._flags = _flags
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

    def _add_task_timing_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            start_time = self._task_ids_to_start_time[task_id]
            if self._enforce_deadlines:
                s.add(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            s.add(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.add(task.deadline - start_time -
                  task.remaining_time == self._task_ids_to_cost[task_id])

    def _add_task_dependency_constraints(self, s):
        for task_id, task in self._task_ids_to_task.items():
            children = self._task_graph.get_children(task)
            for child_task in children:
                # Dependent tasks need to finish before the next one.
                if child_task.id in self._task_ids_to_start_time:
                    s.add(self._task_ids_to_start_time[task_id] +
                          task.remaining_time <= self._task_ids_to_start_time[
                              child_task.id])

    def _add_task_pinning_constraints(self, s):
        if not self._preemptive:
            for task_id, task in self._task_ids_to_task.items():
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(self._task_ids_to_placement[task_id] == 0)

    def _add_task_resource_constraints(self, s, res_type_to_id_range):
        # Add constraints whether a task is placed on GPU or CPU.
        # from num_cpus to num_cpus to num_gpus.
        for task_id, task in self._task_ids_to_task.items():
            assert len(task.resource_requirements
                       ) <= 1, "Doesn't support multi-resource requirements"
            for resource in task.resource_requirements._resource_vector.keys():
                (start_id, end_id) = res_type_to_id_range[resource.name]
                s.add(self._task_ids_to_placement[task_id] >= start_id)
                s.add(self._task_ids_to_placement[task_id] < end_id)

        for t1_id, task1 in self._task_ids_to_task.items():
            for t2_id, task2 in self._task_ids_to_task.items():
                if t2_id >= t1_id:
                    continue
                disjoint = [
                    self._task_ids_to_start_time[t1_id] + task1.remaining_time
                    <= self._task_ids_to_start_time[t2_id],
                    self._task_ids_to_start_time[t2_id] + task2.remaining_time
                    <= self._task_ids_to_start_time[t1_id]
                ]
                s.add(
                    Implies(
                        self._task_ids_to_placement[t1_id] ==
                        self._task_ids_to_placement[t2_id], Or(disjoint)))

    def schedule(self, sim_time: float, task_graph: TaskGraph,
                 worker_pools: WorkerPools):

        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        self._time = sim_time
        self._task_ids_to_task = {}
        self._task_ids_to_start_time = {}
        self._task_ids_to_placement = {}
        self._task_ids_to_cost = {}
        self._task_graph = task_graph
        self._worker_pools = worker_pools

        tasks = task_graph.get_schedulable_tasks(sim_time,
                                                 self.scheduling_horizon,
                                                 self.preemptive, worker_pools)

        scheduler_start_time = time.time()
        if self._goal != 'feasibility':
            s = Optimize()
        else:
            s = Solver()

        for task in tasks:
            self._task_ids_to_task[task.id] = task
            self._task_ids_to_start_time[task.id] = Int(f't{task.id}')
            self._task_ids_to_cost[task.id] = Int(f'c{task.id}')
            self._task_ids_to_placement[task.id] = Int(f'p{task.id}')

        (res_type_to_id_range,
         res_id_to_wp_id) = worker_pools.get_resource_ilp_encoding()
        self._add_task_timing_constraints(s)
        self._add_task_resource_constraints(s, res_type_to_id_range)
        self._add_task_dependency_constraints(s)
        self._add_task_pinning_constraints(s)

        if self._goal != 'feasibility':
            result = s.maximize(sum_costs(self._task_ids_to_cost.values()))

        schedulable = s.check()
        scheduler_end_time = time.time()
        self._runtime = scheduler_end_time - scheduler_start_time\
            if self.runtime == -1 else self.runtime

        # if self._flags.ilp_log_dir is not None:
        #     log_dir = self._flags.ilp_log_dir + f"{self._goal}.smt"
        #     with open(log_dir, "w") as outfile:
        #         outfile.write(s.sexpr())
        #         if self._goal == 'feasibility':
        #             outfile.write("(check-sat)")

        self._cost = sys.maxsize
        self._logger.debug(f"Solver found {schedulable} solution")
        if schedulable != unsat:
            if self._goal != 'feasibility':
                self._cost = s.lower(result)
                self._logger.debug(
                    f"Solver found solution with cost {self._cost}")
            self._placements = []
            for task_id, task in self._task_ids_to_task.items():
                start_time = int(
                    str(s.model()[self._task_ids_to_start_time[task_id]]))
                placement = res_id_to_wp_id[int(
                    str(s.model()[self._task_ids_to_placement[task_id]]))]
                if start_time <= sim_time + self.runtime:
                    # Only places tasks with a start time smaller than the
                    # time at the end of the scheduler run.
                    self._placements.append((task, placement))
                else:
                    self._placements.append((task, None))
            start_times = {}
            for task_id, st_var in self._task_ids_to_start_time.items():
                start_times[task_id] = int(str(s.model()[st_var]))
            self._verify_schedule(self._worker_pools, self._task_graph,
                                  self._placements, start_times)
        else:
            self._placements = [(task, None)
                                for task in self._task_ids_to_task.values()]
        # Log the scheduler run.
        self.log()
        return self.runtime, self._placements

    def log(self):
        if (self._flags is not None
                and self._flags.scheduler_log_file_name is not None):
            with open(self._flags.scheduler_log_file_name, 'wb') as log_file:
                logged_data = {
                    'time': self._time,
                    'tasks': self._task_ids_to_task,
                    'task_graph': self._task_graph,
                    'worker_pools': self._worker_pools,
                    'scheduling_horizon': self._scheduling_horizon,
                    'runtime': self.runtime,
                    'placements': self._placements,
                    'cost': self._cost
                }
                pickle.dump(logged_data, log_file)

    @property
    def preemptive(self):
        return self._preemptive

    @property
    def runtime(self):
        return self._runtime

    @property
    def scheduling_horizon(self):
        return self._scheduling_horizon
