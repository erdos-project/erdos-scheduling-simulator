import functools
from z3 import Int, Solver, Implies, Or, unsat, Optimize, IntVal
# from z3 import If, BitVec, Bool, UGE, ULT, BitVecVal
import logging
import time

from math import ceil, log2
from typing import Dict, List, Optional

import utils
from schedulers.ilp_scheduler import ILPScheduler


class Z3Scheduler(ILPScheduler):

    def __init__(self,
                 preemptive: bool = False,
                 runtime: float = -1.0,
                 _logger: Optional[logging.Logger] = None):
        self._preemptive = preemptive
        self._runtime = runtime
        # Set up the logger.
        if _logger:
            self._logger = _logger
        else:
            self._logger = utils.setup_logging(name="z3")

    def schedule(self,
                 resource_requirement: List[List[bool]],
                 release_times: List[int],
                 absolute_deadlines: List[int],
                 expected_runtimes: List[int],
                 dependency_matrix,
                 pinned_tasks: List[int],
                 num_tasks: int,
                 num_resources: Dict[str, int],
                 bits=None,
                 goal='max_slack',
                 enforce_deadline=True,
                 log_dir=None,
                 logger: Optional[logging.Logger] = None):
        """Runs scheduling using Z3.

        Args:
            resource_requirement: List[List[bool]], List of lists of booleans,
                where each sublist is a list of booleans indicating whether a
                task conforms to a resource type.
            release_times (`List[int]`): List of ints, one for each task,
                indicating the release time of the task.
            absolute_deadlines (`List[int]`): List of ints, one for each task,
                indicating the absolute deadline of the task.
            expected_runtimes (`List[int]`): List of ints, one for each task,
                indicating the expected runtime of the task.
            dependency_matrix (`List[List[bool]]`): List of lists of booleans,
                one for each task, indicating whether task i must finish
                before task j starts.
            pinned_tasks (`List[int]`): List of ints, one for each task,
                indicating the hardware index if a task is pinned to that
                resource (or already running there).
            num_tasks (`int`): Number of tasks.
            num_resources (`List[int]`): Number of resources.
            bits (`int`): Number of bits to use for the ILP.
            goal (`str`): Goal of the scheduler run.
            enforce_deadline (`bool`): Deadline must be met or else the
                schedule will return None.
            log_dir (`str`): Directory to write the ILP to.
            logger(`Optional[logging.Logger]`): The logger to use to log the
                results of the execution.
        """
        assert (len(num_resources) == 2), "Only support 2 resources"
        num_cpus = num_resources[0]
        num_gpus = num_resources[1]

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
        times = [Int(f't{i}')
                 for i in range(0, num_tasks)]  # Time when execution starts
        costs = [Int(f'c{i}') for i in range(0, num_tasks)]  # Costs of gap
        placements = [Int(f'p{i}')
                      for i in range(0, num_tasks)]  # placement on CPU or GPU

        if goal != 'feasibility':
            s = Optimize()
        else:
            s = Solver()

        # Add constraints
        for i, (t, r, e, d, c) in enumerate(
                zip(times, release_times, expected_runtimes,
                    absolute_deadlines, costs)):
            # Finish before deadline.
            if enforce_deadline:
                s.add(t + e <= d)
            # Start at or after release time.
            s.add(r <= t)
            s.add(c == d - e - t)

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
                    times[row_i] + expected_runtimes[row_i] <= times[col_j],
                    times[col_j] + expected_runtimes[col_j] <= times[row_i]
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

        for i, pin in enumerate(pinned_tasks):
            if pin:
                s.add(placements[i] == IntVal(pin))
                # TODO: wrong

        if goal != 'feasibility':
            result = s.maximize(MySum(costs))

        if log_dir is not None:
            log_dir = log_dir + f"{goal}.smt"
            with open(log_dir, "w") as outfile:
                outfile.write(s.sexpr())
                if goal == 'feasibility':
                    outfile.write("(check-sat)")

        schedulable = s.check()
        end_time = time.time()
        runtime = end_time - start_time
        cost = None
        if goal != 'feasibility':
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
