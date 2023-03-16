import time
from collections import defaultdict
from typing import Mapping, Optional, Sequence

import absl  # noqa: F401
import z3

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPools
from workload import (
    BranchPredictionPolicy,
    Placement,
    Placements,
    Resource,
    Task,
    Workload,
)

DEADLINE_ACHIEVEMENT_WEIGHT = 10000000
TASK_SKIP_PENALTY = -2000000000


class TaskOptimizerVariables:
    """This class represents the optimizer variables to be used for each of the task
    that need to be scheduled in a particular run of the Scheduler.

    Args:
        current_time (`EventTime`): The time at which the scheduler was invoked.
        task (`Task`): The task for which the variables are to be constructed.
        workers (`Mapping[int, Worker]`): A shared mapping of an index to the
            Workers available at this scheduling run.
        optimizer (`z3.z3.Optimize`): The optimizer instance to add the variables to.
        enforce_deadlines (`bool`): If `True`, the variables are forced to enforce
            deadlines.
    """

    def __init__(
        self,
        current_time: EventTime,
        task: Task,
        workers: Mapping[int, Worker],
        optimizer: z3.z3.Optimize,
        enforce_deadlines: bool = True,
    ):
        self._task = task

        # Timing characteristics.
        self._start_time = z3.Int(f"{task.unique_name}_start")

        # Placement characteristics.
        self._is_placed = z3.Bool(f"{task.unique_name}_is_placed")
        self._placed_on_worker = z3.BitVec(f"{task.unique_name}_worker", len(workers))

        # If any worker can accomodate the task, find the optimal ones.
        if any(worker.can_accomodate_task(task) for worker in workers.values()):
            resource_types = set(
                resource.name for resource, _ in task.resource_requirements.resources
            )
            self._resources = {
                resource_type: z3.BitVec(
                    f"{task.unique_name}_{resource_type}",
                    max(
                        worker.resources.get_available_quantity(
                            Resource(name=resource_type, _id="any")
                        )
                        for worker in workers.values()
                    ),
                )
                for resource_type in resource_types
            }

            # Initialize the constraints.
            self.initialize_constraints(
                current_time, optimizer, workers, enforce_deadlines
            )
        else:
            # Force the placement to be False.
            optimizer.add(self._is_placed == False)  # noqa: E712
            self._resources = None

    @property
    def start_time(self):
        return self._start_time

    @property
    def task(self):
        return self._task

    @property
    def name(self):
        return self._task.unique_name

    @property
    def is_placed(self):
        return self._is_placed

    @property
    def placed_on_worker(self):
        return self._placed_on_worker

    @property
    def resources(self):
        return self._resources

    def _initialize_timing_constraints(
        self, current_time: EventTime, optimizer, enforce_deadlines: bool = True
    ):
        # Add a constraint to bound the start time be atleast the intended release
        # time of the task.
        optimizer.add(
            z3.And(
                self.start_time >= self.task.release_time.to(EventTime.Unit.US).time,
                self.start_time >= current_time.to(EventTime.Unit.US).time,
            )
        )

        # If requested, add a constraint to ensure that deadlines are met.
        if enforce_deadlines:
            # Ensure that the deadlines are met.
            optimizer.add_soft(
                self.start_time + self.task.remaining_time.to(EventTime.Unit.US).time
                <= self.task.deadline.to(EventTime.Unit.US).time,
            )
        else:
            # Add a soft constraint to ensure the achievement of the deadline.
            # Also, tell Z3 to prefer placing tasks.
            optimizer.add_soft(
                self.start_time + self.task.remaining_time.to(EventTime.Unit.US).time
                <= self.task.deadline.to(EventTime.Unit.US).time,
                weight=DEADLINE_ACHIEVEMENT_WEIGHT,
            )
            optimizer.add_soft(
                self.is_placed == True, weight=DEADLINE_ACHIEVEMENT_WEIGHT  # noqa: E712
            )

    def _initialize_placement_constraints(self, optimizer, workers):
        # Add a constraint to ensure that the task is only placed on a single worker.
        # We model each worker as a single bit of the bitvector, and the following
        # constraint ensures that at most only a single bit of the vector is turned
        # on. If no bit is turned on, the task was not placed.
        num_workers = len(workers)
        optimizer.add(
            z3.Or(
                [
                    self.placed_on_worker == ((2 ** (num_workers - 1)) >> i)
                    for i in range(num_workers + 1)
                ]
            )
        )

        # Add a constraint to ensure that the is_placed is True iff the task is not
        # placed on any worker (i.e., no bit of placed_on_worker was turned on).
        optimizer.add(self.is_placed == (self.placed_on_worker != 0))

    def _initialize_resource_constraints(self, optimizer, workers):
        # Add a constraint to ensure that if the task is placed on a worker that has
        # fewer than the maximum quantity of the resource type, then assume that the
        # first n bits of the vector are already allocated to a phantom task, and the
        # last m bits have the allocation that we require.
        for index, worker in workers.items():
            # Check that the worker has enough available resources for this task to
            # be placed on it. If not, inform z3 to never place this task on the worker.
            can_be_placed = True
            for resource_name in self.resources:
                resource = Resource(name=resource_name, _id="any")
                num_resource_on_worker = worker.resources.get_available_quantity(
                    resource
                )
                num_required_by_task = (
                    self.task.resource_requirements.get_total_quantity(resource)
                )
                if num_resource_on_worker < num_required_by_task:
                    can_be_placed = False
                    break

            if can_be_placed:
                # If the task can be placed, ensure the allocation of unavailable
                # resources to phantom tasks and the allocation of appropriate quantity
                # of resources as required by the task.
                for resource_name, variable in self.resources.items():
                    resource = Resource(name=resource_name, _id="any")
                    num_required_by_task = (
                        self.task.resource_requirements.get_total_quantity(resource)
                    )
                    bottom_m_bits = worker.resources.get_available_quantity(resource)
                    top_n_bits = variable.size() - bottom_m_bits
                    assert top_n_bits >= 0 and bottom_m_bits >= 0, (
                        "Negative quantity of top or bottom bits "
                        "received in resource allocation."
                    )

                    # Ensure that the bottom m bits have the quantity we require, and
                    # the top n bits are set to 1.
                    allowed_values = [
                        int(
                            "1" * top_n_bits + format(val, f"#0{bottom_m_bits+2}b")[2:],
                            base=2,
                        )
                        for val in range(2**bottom_m_bits)
                        if sum(map(int, bin(val)[2:])) == num_required_by_task
                    ]
                    optimizer.add(
                        z3.Implies(
                            self.placed_on_worker == index,
                            z3.Or([variable == value for value in allowed_values]),
                        )
                    )
            else:
                # If the task cannot be placed, hint the z3 optimizer to never place
                # the task here.
                optimizer.add(
                    z3.Implies(self.is_placed, self.placed_on_worker != index)
                )

    def initialize_constraints(
        self,
        current_time: EventTime,
        optimizer: z3.z3.Optimize,
        workers: Mapping[int, Worker],
        enforce_deadlines: bool = True,
    ):
        self._initialize_timing_constraints(current_time, optimizer, enforce_deadlines)
        self._initialize_placement_constraints(optimizer, workers)
        self._initialize_resource_constraints(optimizer, workers)

    def __str__(self):
        return f"OptVars(name={self.name})"

    def __repr__(self):
        return str(self)


class Z3Scheduler(BaseScheduler):
    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime(-1, EventTime.Unit.US),
        lookahead: int = EventTime.zero(),
        enforce_deadlines: bool = False,
        policy: BranchPredictionPolicy = BranchPredictionPolicy.RANDOM,
        branch_prediction_accuracy: float = 0.50,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        goal: str = "max_slack",
        _flags: Optional["absl.flags"] = None,
    ):
        super(Z3Scheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            policy=policy,
            branch_prediction_accuracy=branch_prediction_accuracy,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        self._goal = goal

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        print(sim_time)
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled = workload.get_schedulable_tasks(
            sim_time,
            self.lookahead,
            self.preemptive,
            self.retract_schedules,
            worker_pools,
            self.policy,
            self.branch_prediction_accuracy,
            self.release_taskgraphs,
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received "
            f"{[task.unique_name for task in tasks_to_be_scheduled]} "
            f"tasks for scheduling."
        )

        # TODO (Sukrit): Reconstruct a worker pool based on the preemptive nature of
        # the scheduler.

        # Construct a mapping of the index of a worker to the worker itself.
        # The index of the worker is defined by the bit that is turned on in a
        # BitVector with a size equal to the number of workers in the system.
        worker_index = 0
        workers = {}
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                workers[2**worker_index] = worker
                worker_index += 1

        # Construct a mapping from the worker to the WorkerPool to which it belongs.
        worker_to_worker_pool = {}
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                worker_to_worker_pool[worker.id] = worker_pool.id

        # Construct the Optimizer, generate the variables and add constraints.
        scheduler_start_time = time.time()
        optimizer = z3.Optimize()
        tasks_to_variables = self._add_variables(
            sim_time, optimizer, tasks_to_be_scheduled, workers
        )
        self._add_task_dependency_constraints(optimizer, tasks_to_variables, workload)
        self._add_resource_constraints(optimizer, tasks_to_variables, workload, workers)

        # Add the objectives, and return the results.
        goal = self._add_objective(optimizer, tasks_to_variables, workload)
        placements = []
        self._logger.debug(
            f"[{sim_time.time}] The scheduler returned: {optimizer.check()}."
        )
        if optimizer.check() == z3.sat:
            model = optimizer.model()
            self._logger.info(
                f"[{sim_time.time}] The optimized goal value was {model[goal]}."
            )
            model_str = "\n".join([f"{var}: {model[var]}" for var in list(model)])
            self._logger.debug(f"[{sim_time.time}] The model was \n {model_str}")
            for task_name, variables in tasks_to_variables.items():
                if variables.is_placed is not None and model[variables.is_placed]:
                    worker = workers[model[variables.placed_on_worker].as_long()]
                    worker_pool_id = worker_to_worker_pool[worker.id]
                    start_time = model[variables.start_time].as_long()
                    placements.append(
                        Placement(
                            task=variables.task,
                            placement_time=EventTime(start_time, EventTime.Unit.US),
                            worker_pool_id=worker_pool_id,
                            worker_id=worker.id,
                        )
                    )
                    self._logger.debug(
                        f"[{sim_time.time}] Placed {variables.task} on "
                        f"WorkerPool({worker_pool_id}) to be started at {start_time}."
                    )
                else:
                    placements.append(Placement(variables.task))
                    self._logger.debug(
                        f"[{sim_time.time}] Failed to place {variables.task} because "
                        f"no worker pool could accomodate the resource requirements."
                    )
        else:
            for variables in tasks_to_variables.values():
                placements.append(Placement(variables.task))
            self._logger.debug(f"[{sim_time.time}] Failed to place any task.")
        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        self._logger.debug(
            f"[{sim_time.time}] The runtime of the scheduler was: {scheduler_runtime}."
        )
        runtime = (
            scheduler_runtime
            if self.runtime == EventTime(-1, EventTime.Unit.US)
            else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )

    def _add_variables(
        self,
        sim_time: EventTime,
        optimizer: z3.z3.Optimize,
        tasks_to_be_scheduled: Sequence[Task],
        workers: Mapping[int, Worker],
    ) -> Mapping[str, TaskOptimizerVariables]:
        tasks_to_variables = {}
        for task in tasks_to_be_scheduled:
            tasks_to_variables[task.unique_name] = TaskOptimizerVariables(
                sim_time, task, workers, optimizer, self.enforce_deadlines
            )
        return tasks_to_variables

    def _add_task_dependency_constraints(self, optimizer, tasks_to_variables, workload):
        for _, variables in tasks_to_variables.items():
            task = variables.task
            task_graph = workload.get_task_graph(variables.task.task_graph)
            parent_variables = [
                tasks_to_variables[parent.unique_name]
                for parent in task_graph.get_parents(task)
                if parent.unique_name in tasks_to_variables
            ]
            # Ensure that the task is placed, only if all of its parents were placed.
            optimizer.add(
                z3.Implies(
                    variables.is_placed,
                    z3.And([parent.is_placed for parent in parent_variables]),
                )
            )

            # Ensure that if the task is placed, that it is only started once all
            # of its parents are finished.
            optimizer.add(
                z3.Implies(
                    variables.is_placed,
                    z3.And(
                        [
                            variables.start_time
                            >= parent.start_time
                            + parent.task.remaining_time.to(EventTime.Unit.US).time
                            for parent in parent_variables
                        ]
                    ),
                )
            )

    def _add_resource_constraints(
        self, optimizer, tasks_to_variables, workload, workers
    ):
        # Find all tasks that might run in parallel with each task.
        task_resource_dependencies = defaultdict(set)
        for task_name_1, variable_1 in tasks_to_variables.items():
            for task_name_2, variable_2 in tasks_to_variables.items():
                if (
                    task_name_1 == task_name_2
                    or task_name_1 in task_resource_dependencies[task_name_2]
                    or variable_1.resources is None
                    or variable_2.resources is None
                ):
                    # Either same task or we've already added this pair's constraints,
                    # or one of the tasks cannot be placed.
                    continue

                if variable_1.task.task_graph == variable_2.task.task_graph:
                    # If the tasks are in the same graph, and they are not dependent
                    # on each other, they can run in parallel, and hence, must be
                    # checked for overlaps.
                    task_graph = workload.get_task_graph(variable_1.task.task_graph)
                    if not task_graph.are_dependent(variable_1.task, variable_2.task):
                        task_resource_dependencies[task_name_1].add(task_name_2)
                else:
                    # If the tasks belong to different graphs, they can always run in
                    # parallel, and hence, must be checked for overlaps.
                    task_resource_dependencies[task_name_1].add(task_name_2)

        for index, worker in workers.items():
            # For each of the task and its potential parallelly executable tasks,
            # if any two of them are executed on the same worker and potentially
            # overlap, ensure that they do not occupy the same resource index for any
            # of their resource requirements.
            for task_name, dependencies in task_resource_dependencies.items():
                for dependency_name in dependencies:
                    # The variables for each task.
                    variable_task_1 = tasks_to_variables[task_name]
                    variable_task_2 = tasks_to_variables[dependency_name]

                    task_1 = variable_task_1.task
                    task_2 = variable_task_2.task

                    # Add a variable to check the overlap of the two tasks.
                    # If the first or the second task ends before the other starts,
                    # then we define them to be non-overlapping.
                    first_ends_before_second_starts = z3.Bool(
                        f"{task_name}_ends_before_{dependency_name}_starts"
                    )
                    optimizer.add(
                        first_ends_before_second_starts
                        == (
                            (
                                variable_task_1.start_time
                                + task_1.remaining_time.to(EventTime.Unit.US).time
                            )
                            < variable_task_2.start_time
                        )
                    )

                    second_ends_before_first_starts = z3.Bool(
                        f"{dependency_name}_ends_before_{task_name}_starts"
                    )
                    optimizer.add(
                        second_ends_before_first_starts
                        == (
                            (
                                variable_task_2.start_time
                                + task_2.remaining_time.to(EventTime.Unit.US).time
                            )
                            < variable_task_1.start_time
                        )
                    )

                    overlap_variable = z3.Bool(f"{task_name}_{dependency_name}_overlap")
                    optimizer.add(
                        overlap_variable
                        == z3.Not(
                            z3.Or(
                                # Either the first task ends before the
                                # second starts.
                                first_ends_before_second_starts,
                                # Or the second one ends before the first
                                # starts.
                                second_ends_before_first_starts,
                            )
                        )
                    )

                    # If they are placed on the same worker, and may overlap in their
                    # execution, ensure that the XOR of their resource allocations is 0
                    # for all the resources used by both of the tasks.
                    resource_check_variables = []
                    for resource, quantity in worker.resources.resources:
                        if (
                            resource.name in variable_task_1.resources
                            and resource.name in variable_task_2.resources
                        ):
                            worker_resource_variable = z3.Bool(
                                f"{worker.name}_{resource.name}_independent_"
                                f"{task_name}_{dependency_name}"
                            )
                            optimizer.add(
                                worker_resource_variable
                                == (
                                    z3.Extract(
                                        quantity - 1,
                                        0,
                                        variable_task_1.resources[resource.name],
                                    )
                                    ^ z3.Extract(
                                        quantity - 1,
                                        0,
                                        variable_task_2.resources[resource.name],
                                    )
                                    == int("1" * quantity, base=2)
                                )
                            )
                            resource_check_variables.append(worker_resource_variable)

                    # Add a check to ensure that if the two tasks are placed on the
                    # current worker, and they overlap in their execution, then
                    # the resource indices are exclusively assigned.
                    optimizer.add(
                        z3.Implies(
                            z3.And(
                                variable_task_1.is_placed,
                                variable_task_2.is_placed,
                                variable_task_1.placed_on_worker == index,
                                variable_task_2.placed_on_worker == index,
                                overlap_variable,
                            ),
                            z3.And(resource_check_variables),
                        )
                    )

    def _add_objective(self, optimizer, tasks_to_variables, workload: Workload):
        goal = None
        if self._goal == "max_slack":
            # Add the penalty as a z3 constant.
            task_skip_penalty = z3.Int("TASK_SKIP_PENALTY")
            optimizer.add(task_skip_penalty == TASK_SKIP_PENALTY)

            # Find the TaskGraphs and their corresponding tasks.
            task_graph_to_tasks = defaultdict(list)
            for variable in tasks_to_variables.values():
                task_graph_to_tasks[variable.task.task_graph].append(variable.task)

            # Find the slack for each of the TaskGraph.
            task_graph_to_reward = {}
            for task_graph_name, tasks in task_graph_to_tasks.items():
                # TODO (Sukrit): This check might fail if the last node
                # is not in the schedulable tasks, and the arbitrary
                # chosen node on the same level is allocated an earlier
                # start time than the other ones.
                task_graph = workload.get_task_graph(task_graph_name)
                last_task, last_task_depth = tasks[0], task_graph.get_node_depth(
                    tasks[0]
                )
                for task in tasks[1:]:
                    if task_graph.get_node_depth(task) > last_task_depth:
                        last_task = task
                        last_task_depth = task_graph.get_node_depth(task)
                print(f"Last task for {task_graph_name} is {last_task.unique_name}.")

                last_task_variable = tasks_to_variables[last_task.unique_name]
                task_graph_slack = z3.Int(f"{task_graph_name}_slack")
                optimizer.add(
                    task_graph_slack
                    == (
                        task_graph.deadline.to(EventTime.Unit.US).time
                        - last_task.remaining_time.to(EventTime.Unit.US).time
                        - last_task_variable.start_time
                    )
                )
                task_graph_to_reward[task_graph_name] = task_graph_slack

            # Slack is defined as the deadline - remaining_time - start_time
            total_slack = []
            for variable in tasks_to_variables.values():
                # task_graph = workload.get_task_graph(variable.task.task_graph)
                # task_slack = z3.Int(f"{variable.task.unique_name}_slack")
                # optimizer.add(
                #     task_slack
                #     == (
                #         task_graph.deadline.to(EventTime.Unit.US).time
                #         - task_graph.get_remaining_time(self._policy)
                #         .to(EventTime.Unit.US)
                #         .time
                #         - variable.start_time
                #     )
                # )
                total_slack.append(
                    z3.If(
                        variable.is_placed,
                        task_graph_to_reward[variable.task.task_graph],
                        task_skip_penalty,
                    )
                )
            goal = z3.Int("TASK_SLACK_SUM")
            optimizer.add(goal == z3.Sum(total_slack))
            optimizer.maximize(goal)
        else:
            raise ValueError(f"Goal {self._goal} not supported yet.")
        return goal
