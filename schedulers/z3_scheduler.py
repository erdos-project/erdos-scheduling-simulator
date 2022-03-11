import functools
from z3 import Int, Solver, Implies, Or, unsat, Optimize
# from z3 import If, BitVec, Bool, UGE, ULT, BitVecVal
import time

# from math import ceil, log2
from typing import Optional, Sequence

import utils
from schedulers import BaseScheduler
from workload import Resource, Resources, Task, TaskGraph, TaskState
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
        # Set up the logger.
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
                    start_times[row_i] + tasks[row_i].runtime <=
                    start_times[col_j],
                    start_times[col_j] + tasks[col_j].runtime <=
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
            for r_name, (start_id, end_id) in res_type_to_id_range.items():
                if task.resource_requirements >= Resources(
                    {Resource(name=r_name, _id="any"): 1}):
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
                 task_graph: TaskGraph, wps: WorkerPools):

        def sum_costs(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        tasks = self._get_schedulable_tasks(released_tasks, wps)

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
                s.addConstr(start_time + task.runtime <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= start_time)
            # Defines cost as slack deadline - finish time.
            s.addConstr(task.deadline - start_time - task.runtime == cost_var)

        (res_type_to_id_range,
         res_id_to_wp_id) = wps.resource_type_id_range_map()
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
        runtime = scheduler_end_time - scheduler_start_time\
            if self.runtime == -1 else self.runtime

        if self._flags.ilp_log_dir is not None:
            log_dir = self._flags.ilp_log_dir + f"{self._goal}.smt"
            with open(log_dir, "w") as outfile:
                outfile.write(s.sexpr())
                if self._goal == 'feasibility':
                    outfile.write("(check-sat)")

        cost = None
        if self._goal != 'feasibility':
            cost = s.lower(result)
            self._logger.debug(f"Solver found solution with cost {cost}")
        self._logger.debug(f"Solver found {schedulable} solution")
        if schedulable != unsat:
            # placements = [
            #     int(str(s.model()[p])) if not needs_gpu[i] else
            #     (10 if bool(s.model()[p]) else 11)
            #     for i, p in enumerate(placements)
            # ]
            for task, start_time_v, placement_v in zip(tasks, start_times,
                                                       placements):
                start_time = int(str(s.model()[start_time_v]))
                placement = int(str(s.model()[placement_v]))
                # TODO: It only places tasks with a start time smaller
                # than the time at the end of the scheduler run.
                if int(start_time) <= sim_time + runtime:
                    result.append((task, res_id_to_wp_id[placement]))
                else:
                    result.append((task, None))
            return (runtime, result)
        return runtime, None
