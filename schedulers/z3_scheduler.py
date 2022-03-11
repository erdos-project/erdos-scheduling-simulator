import functools
from z3 import Int, Solver, Implies, Or, unsat, Optimize, IntVal
# from z3 import If, BitVec, Bool, UGE, ULT, BitVecVal
import time

# from math import ceil, log2
from typing import Dict, List, Optional

import utils
from schedulers.ilp_scheduler import ILPScheduler
from workload import Task, TaskState


class Z3Scheduler(ILPScheduler):

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

    def schedule(self, resource_requirement: List[List[bool]],
                 tasks: List[Task], dependency_matrix,
                 num_resources: Dict[str, int]):
        """Runs scheduling using Z3.

        Args:
            resource_requirement: List[List[bool]], List of lists of booleans,
                where each sublist is a list of booleans indicating whether a
                task conforms to a resource type.
            dependency_matrix (`List[List[bool]]`): List of lists of booleans,
                one for each task, indicating whether task i must finish
                before task j starts.
            num_resources (`List[int]`): Number of resources.
        """
        num_resources_ub = [
            sum(num_resources[0:i + 1]) for i in range(len(num_resources))
        ]
        num_resources_lb = [0] + [
            sum(num_resources[0:i + 1]) for i in range(len(num_resources) - 1)
        ]

        def safe_resource_index(resource_requirement: List[bool]):
            # check that there's only one True
            compats = [r for r in resource_requirement if r]
            if len(compats) > 1:
                raise Exception("More than one compatible resource")
            if len(compats) == 0:
                raise Exception("No compatible resource")
            return resource_requirement.index(True)

        def MySum(lst):
            return functools.reduce(lambda a, b: a + b, lst, 0)

        # Add new constraint node 'new_node' with attribute 'attr'
        # e.g. call add_relation(G, 'ub_a', {'rel:'<', 'num':5}, ['a'])
        #  to add constraint (a < 5)

        # bits = ceil(log2(num_gpus + num_cpus))

        start_time = time.time()
        # Time when execution starts.
        times = [Int(f't{i}') for i in range(0, len(tasks))]
        # Costs of gap
        costs = [Int(f'c{i}') for i in range(0, len(tasks))]
        # Placement on CPU or GPU.
        placements = [Int(f'p{i}') for i in range(0, len(tasks))]

        if self._goal != 'feasibility':
            s = Optimize()
        else:
            s = Solver()

        # Add constraints
        for time_var, task, cost_var in zip(times, tasks, costs):
            if self._enforce_deadlines:
                s.addConstr(time_var + task.runtime <= task.deadline)
            # Start at or after release time.
            s.addConstr(task.release_time <= time_var)
            # Defines cost as slack deadline - finish time.
            s.addConstr(task.deadline - time_var - task.runtime == cost_var)

        for i, (p, res_req) in enumerate(zip(placements,
                                             resource_requirement)):
            resource_index = safe_resource_index(
                res_req)  # convert bool vec to int index
            s.add(p >= num_resources_lb[resource_index],
                  p < num_resources_ub[resource_index])

            # Legacy code for converting to bitvec to speed up solving if the
            # number of a particular resource is small.
            # placements[i] = Bool(f'p{i}')
            # gpu_bits = ceil(log2(num_gpus))
            # placements[i] = BitVecVal(num_cpus, gpu_bits)
            # s.add(UGE(p , BitVecVal(num_cpus, gpu_bits)), ULT(p ,
            #       BitVecVal(num_gpus + num_cpus, gpu_bits)))
            # elif res_req == [False,True]:
            #     s.add(p >= num_resources_lb, p < num_resources_ub)
            # cpu_bits = ceil(log2(num_cpus))
            # placements[i] = BitVecVal(num_cpus, cpu_bits)
            # First half is num_cpus.
            # s.add(UGE(p, BitVecVal(0, cpu_bits)),
            #       ULT (p , BitVecVal(num_cpus, cpu_bits)))
            # else:
            #     raise ValueError("Tasks are only allowed on one resource")

        for row_i in range(len(dependency_matrix)):
            for col_j in range(len(dependency_matrix[0])):
                disjoint = [
                    times[row_i] + tasks[row_i].runtime <= times[col_j],
                    times[col_j] + tasks[col_j].runtime <= times[row_i]
                ]
                if dependency_matrix[row_i][col_j]:
                    s.add(
                        disjoint[0]
                    )  # dependent jobs need to finish before the next one

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

        if not self._preemptive:
            for i, task in enumerate(tasks):
                if task.state == TaskState.RUNNING:
                    # TODO: Incorrect! Placement must equate to resource id!
                    s.add(placements[i] == 0)

        if self._goal != 'feasibility':
            result = s.maximize(MySum(costs))

        if self._flags.ilp_log_dir is not None:
            log_dir = self._flags.ilp_log_dir + f"{self._goal}.smt"
            with open(log_dir, "w") as outfile:
                outfile.write(s.sexpr())
                if self._goal == 'feasibility':
                    outfile.write("(check-sat)")

        schedulable = s.check()
        end_time = time.time()
        runtime = end_time - start_time
        cost = None
        if self._goal != 'feasibility':
            cost = s.lower(result)
            self._logger.debug(cost)
        self._logger.debug(schedulable)
        if schedulable != unsat:
            # placements = [
            #     int(str(s.model()[p])) if not needs_gpu[i] else
            #     (10 if bool(s.model()[p]) else 11)
            #     for i, p in enumerate(placements)
            # ]
            placements = [
                int(str(s.model()[p])) for i, p in enumerate(placements)
            ]

            outputs = [int(str(s.model()[t]))
                       for t in times], [p for p in placements]
            self._logger.debug(outputs)
            return outputs, cost, runtime
        return (None, None), None, runtime
