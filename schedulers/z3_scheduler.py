import functools
import pickle
import time
import sys

import absl  # noqa: F401

# from math import ceil, log2
from typing import Optional, Sequence

from z3 import Int, Solver, Implies, Or, unsat, Optimize
# from z3 import If, BitVec, Bool, UGE, ULT, BitVecVal

import utils
from schedulers import BaseScheduler
from workload import Task, TaskGraph, TaskState
from workers import WorkerPools


class Z3Scheduler(BaseScheduler):

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 goal: str = 'max_slack',
                 enforce_deadlines: bool = True,
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
        self._time = None
        self._released_tasks = None
        self._task_graph = None
        self._worker_pools = None
        self._placements = None
        self._cost = None
        # Set up the logger.
        self._flags = _flags
        if _flags:
            self._logger = utils.setup_logging(name=self.__class__.__name__,
                                               log_file=_flags.log_file_name,
                                               log_level=_flags.log_level)
        else:
            self._logger = utils.setup_logging(name=self.__class__.__name__)

    def _add_task_dependency_constraints(self, s, tasks, dependency_matrix,
                                         start_times, placements):
        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [
                    start_times[row_i] + tasks[row_i].remaining_time <=
                    start_times[col_j],
                    start_times[col_j] + tasks[col_j].remaining_time <=
                    start_times[row_i]
                ]
                # Dependent jobs need to finish before the next one.
                if dependency_matrix[row_i][col_j]:
                    s.add(disjoint[0])

                if row_i != col_j:  # and row_i < col_j ?
                    # if needs_gpu[row_i]:
                    #     i_val = If(placements[row_i],IntVal(10),IntVal(11))
                    # else:
                    i_val = placements[row_i]
                    # if needs_gpu[col_j]:
                    #     j_val = If(placements[col_j],IntVal(10),IntVal(11))
                    # else:
                    j_val = placements[col_j]
                    # Cannot overlap if on the same hardware.
                    # ITE(placements[row_i],10,11)) == \
                    #    ITE(placements[col_j], 10,11))
                    # Implies(placements[row_i] == placements[col_j],
                    #         Or(disjoint))
                    s.add(Implies(i_val == j_val, Or(disjoint)))

    def _add_task_pinning_constraints(self, s, tasks, placements):
        if not self._preemptive:
            for i, task in enumerate(tasks):
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(placements[i] == 0)

    def _add_task_resource_constraints(self, s, tasks, res_type_to_id_range,
                                       placements):
        # Add constraints whether a task is placed on GPU or CPU.
        # from num_cpus to num_cpus to num_gpus.

        for placement, task in zip(placements, tasks):
            assert len(task.resource_requirements
                       ) <= 1, "Doesn't support multi-resource requirements"
            for resource in task.resource_requirements._resource_vector.keys():
                (start_id, end_id) = res_type_to_id_range[resource.name]
                s.add(placement >= start_id)
                s.add(placement < end_id)
            # Legacy code for converting to bitvec to speed up solving if the
            # number of a particular resource is small.
            # placements[i] = Bool(f'p{i}')
            # gpu_bits = ceil(log2(num_gpus))
            # placements[i] = BitVecVal(num_cpus, gpu_bits)
            # s.add(UGE(p , BitVecVal(num_cpus, gpu_bits)), ULT(p ,
            #       BitVecVal(num_gpus + num_cpus, gpu_bits)))
            # elif res_req == [False,True]:
            #     s.add(p >= start_id, p < end_id)
            # cpu_bits = ceil(log2(num_cpus))
            # placements[i] = BitVecVal(num_cpus, cpu_bits)
            # First half is num_cpus.
            # s.add(UGE(p, BitVecVal(0, cpu_bits)),
            #       ULT (p , BitVecVal(num_cpus, cpu_bits)))
            # else:
            #     raise ValueError("Tasks are only allowed on one resource")

    def _get_schedulable_tasks(self, released_tasks, wps):
        # Create the tasks to be scheduled.
        if self.preemptive:
            # Collect all the currently placed tasks on the WorkerPools, along
            # with the set of released tasks.
            tasks = [task for task in released_tasks]
            tasks.extend(wps.get_placed_tasks())
        else:
            # Collect the currently released tasks.
            tasks = [task for task in released_tasks]
        return tasks

    def schedule(self, sim_time: float, released_tasks: Sequence[Task],
                 task_graph: TaskGraph, worker_pools: WorkerPools):

        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        self._time = sim_time
        self._released_tasks = released_tasks
        self._task_graph = task_graph
        self._worker_pools = worker_pools
        # TODO: We should get tasks that will be released later as well.
        tasks = self._get_schedulable_tasks(released_tasks, worker_pools)

        scheduler_start_time = time.time()
        if self._goal != 'feasibility':
            s = Optimize()
        else:
            s = Solver()

        # Time when execution starts.
        start_times = [Int(f't{i}') for i in range(0, len(tasks))]
        # Costs of gap
        costs = [Int(f'c{i}') for i in range(0, len(tasks))]
        # Placement on CPU or GPU.
        placements = [Int(f'p{i}') for i in range(0, len(tasks))]

        # Add constraints
        for start_time, task, cost_var in zip(start_times, tasks, costs):
            if self._enforce_deadlines:
                s.add(start_time + task.remaining_time <= task.deadline)
            # Start at or after release time.
            s.add(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.add(task.deadline - start_time - task.remaining_time == cost_var)

        (res_type_to_id_range,
         res_id_to_wp_id) = worker_pools.get_resource_ilp_encoding()
        self._add_task_resource_constraints(s, tasks, res_type_to_id_range,
                                            placements)
        # TODO(Justin): This doesn't account for the task dependencies.
        dependency_matrix = [[False] * len(tasks)] * len(tasks)
        self._add_task_dependency_constraints(s, tasks, dependency_matrix,
                                              start_times, placements)
        self._add_task_pinning_constraints(s, tasks, placements)

        if self._goal != 'feasibility':
            result = s.maximize(sum_costs(costs))

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
            # placements = [
            #     int(str(s.model()[p])) if not needs_gpu[i] else
            #     (10 if bool(s.model()[p]) else 11)
            #     for i, p in enumerate(placements)
            # ]
            self._placements = []
            start_times = [int(str(s.model()[st])) for st in start_times]
            for task, start_time, placement_v in zip(tasks, start_times,
                                                     placements):
                placement = int(str(s.model()[placement_v]))
                # TODO: It only places tasks with a start time smaller
                # than the time at the end of the scheduler run.
                if start_time <= sim_time + self.runtime:
                    self._placements.append((task, res_id_to_wp_id[placement]))
                else:
                    self._placements.append((task, None))
            self._verify_schedule(self._placements, dependency_matrix,
                                  start_times)
        else:
            self._placements = [(task, None) for task in tasks]
        # Log the scheduler run.
        self.log()
        return self.runtime, self._placements

    def log(self):
        if (self._flags is not None
                and self._flags.scheduler_log_file_name is not None):
            with open(self._flags.scheduler_log_file_name, 'wb') as log_file:
                logged_data = {
                    'time': self._time,
                    'released_tasks': self._released_tasks,
                    'task_graph': self._task_graph,
                    'worker_pools': self._worker_pools,
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
