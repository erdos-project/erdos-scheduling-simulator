import time
from typing import List, Mapping, Optional, Set, Tuple

import absl  # noqa: F401
import tetrisched_py as tetrisched

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    Placement,
    Placements,
    Resource,
    Task,
    TaskGraph,
    TaskState,
    Workload,
)


class TetriSchedScheduler(BaseScheduler):
    """Implements a STRL-based, DAG-aware formulation for the Tetrisched backend.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime.invalid(),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        time_discretization: EventTime = EventTime(1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        if preemptive:
            raise ValueError("TetrischedScheduler does not support preemption.")
        super(TetriSchedScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        self._time_discretization = time_discretization.to(EventTime.Unit.US)
        self._scheduler = tetrisched.Scheduler(
            self._time_discretization.time, tetrisched.backends.SolverBackendType.GUROBI
        )

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled: List[Task] = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
            release_taskgraphs=self.release_taskgraphs,
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks to be scheduled. These tasks along with their deadlines were: "
            f"{[f'{t.unique_name} ({t.deadline})' for t in tasks_to_be_scheduled]}"
        )

        # Construct the STRL expression.
        scheduler_start_time = time.time()
        placements = []
        if len(tasks_to_be_scheduled) > 0 and any(
            task.state != TaskState.SCHEDULED for task in tasks_to_be_scheduled
        ):
            # Construct the partitions from the Workers in the WorkerPool.
            partitions = self.construct_partitions(worker_pools=worker_pools)

            # Construct the STRL expressions for each TaskGraph and add them together
            # in a single objective expression.
            task_graph_names: Set[TaskGraph] = {
                task.task_graph for task in tasks_to_be_scheduled
            }
            objective_strl = tetrisched.strl.ObjectiveExpression(
                f"TetriSched_{sim_time.to(EventTime.Unit.US).time}"
            )
            for task_graph_name in task_graph_names:
                # Retrieve the TaskGraph and construct its STRL.
                task_graph = workload.get_task_graph(task_graph_name)
                task_graph_strl = self.construct_task_graph_strl(
                    current_time=sim_time,
                    task_graph=task_graph,
                    partitions=partitions,
                    task_strls={},
                    tasks_to_be_scheduled=tasks_to_be_scheduled,
                )
                if task_graph_strl is not None:
                    objective_strl.addChild(task_graph_strl)

            # Register the STRL expression with the scheduler and solve it.
            self._scheduler.registerSTRL(objective_strl, partitions, sim_time.time)
            solver_start_time = time.time()
            self._scheduler.schedule(sim_time.time)
            solver_end_time = time.time()
            solver_time = EventTime(
                int((solver_end_time - solver_start_time) * 1e6), EventTime.Unit.US
            )

            # Retrieve the solution and check if we were able to schedule anything.
            solverSolution = objective_strl.getSolution()
            self._logger.info(
                f"[{sim_time.time}] Solver returned utility of {solverSolution.utility}"
                f" and took {solver_time} to solve. The solution result "
                f"was {self._scheduler.getLastSolverSolution()}."
            )
            if solverSolution.utility == 0:
                raise RuntimeError("TetrischedScheduler was unable to schedule tasks.")

            # Retrieve the Placements for each task.
            for task in tasks_to_be_scheduled:
                task_placement = solverSolution.getPlacement(task.unique_name)
                if task_placement is None or not task_placement.isPlaced():
                    self._logger.error(
                        f"[{sim_time.time}] No Placement was found for "
                        f"Task {task.unique_name}."
                    )
                    placements.append(Placement.create_task_placement(task=task))

                # Retrieve the Partition where the task was placed.
                # The task was placed, retrieve the Partition where the task
                # was placed.
                partitionId = task_placement.getPartitionAssignments()[0][0]
                partition = partitions.partitionMap[partitionId]
                task_placement = Placement.create_task_placement(
                    task=task,
                    placement_time=EventTime(
                        task_placement.startTime, EventTime.Unit.US
                    ),
                    worker_id=partition.associatedWorker.id,
                    worker_pool_id=partition.associatedWorkerPool.id,
                    execution_strategy=task.available_execution_strategies[0],
                )
                placements.append(task_placement)
                self._logger.debug(
                    "[%s] Placed %s (with deadline %s and remaining time %s) on "
                    "WorkerPool (%s) to be started at %s and executed with %s.",
                    sim_time.to(EventTime.Unit.US).time,
                    task_placement.task.unique_name,
                    task_placement.task.deadline,
                    task_placement.execution_strategy.runtime,
                    task_placement.worker_pool_id,
                    task_placement.placement_time,
                    task_placement.execution_strategy,
                )

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime if self.runtime == EventTime.invalid() else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )

    def construct_partitions(self, worker_pools: WorkerPools) -> tetrisched.Partitions:
        """Partitions the Workers in the WorkerPools into a granular partition set.

        The Partitions are used to reduce the number of variables in the compiled ILP
        model. All the resources in the Partition are expected to belong to an
        equivalence set and are therefore interchangeable.

        Args:
            worker_pools (`WorkerPools`): The WorkerPools to be partitioned.

        Returns:
            A `Partitions` object that contains the partitions.
        """
        partitions = tetrisched.Partitions()
        # BUG (Sukrit): The partitionMap is being used to keep the Partition objects
        # alive on the Python side so we can query the associatedWorker and the
        # associatedWorkerPool. Otherwise, pybind11 loses track of the objects and
        # the attributes are not accessible.
        partitions.partitionMap = {}
        # TODO (Sukrit): This method constructs a separate partition for all the slots
        # in a Worker. This might not be the best strategy for dealing with heterogenous
        # resources. Fix.
        worker_index = 1
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                # Check that the Worker only has Slot resources.
                for resource, _ in worker.resources.resources:
                    if resource.name != "Slot":
                        raise ValueError(
                            "TetrischedScheduler currently supports Slot resources."
                        )

                # Create a tetrisched Partition.
                slot_quantity = worker.resources.get_total_quantity(
                    resource=Resource(name="Slot", _id="any")
                )
                partition = tetrisched.Partition(
                    worker_index, worker.name, slot_quantity
                )
                partitions.addPartition(partition)

                # Maintain the relevant mappings to transform it to a Placement.
                partition.associatedWorker = worker
                partition.associatedWorkerPool = worker_pool
                partitions.partitionMap[worker_index] = partition
                worker_index += 1
        return partitions

    def construct_task_strl(
        self,
        current_time: EventTime,
        task: Task,
        partitions: tetrisched.Partitions,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The Task for which the STRL expression is to be constructed.
            task_id (`int`): The index of this Task in the Workload.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices as ChooseExpressions and a MaxExpression that selects the best
            placement choice.
        """
        if len(task.available_execution_strategies) > 1:
            raise NotImplementedError(
                "TetrischedScheduler does not support multiple execution strategies."
            )

        # Check that the Task works only on Slots for now.
        # TODO (Sukrit): We should expand this to general resource usage.
        # But, this works for now.
        execution_strategy = task.available_execution_strategies[0]
        for resource, _ in execution_strategy.resources.resources:
            if resource.name != "Slot":
                raise ValueError(
                    "TetrischedScheduler currently only supports Slot resources."
                )

        # Construct the STRL MAX expression for this Task.
        # This enforces the choice of only one placement for this Task.
        self._logger.debug(
            f"[{current_time.time}] Constructing a STRL expression tree for "
            f"{task.name} (runtime={execution_strategy.runtime}, "
            f"deadline={task.deadline}) with name: {task.unique_name}_placement."
        )
        chooseOneFromSet = tetrisched.strl.MaxExpression(
            f"{task.unique_name}_placement"
        )

        # Construct the STRL ChooseExpressions for this Task.
        # This expression represents a particular placement choice for this Task.
        num_slots_required = execution_strategy.resources.get_total_quantity(
            resource=Resource(name="Slot", _id="any")
        )

        num_choose_expressions_generated = 0
        for placement_time in range(
            current_time.to(EventTime.Unit.US).time,
            task.deadline.to(EventTime.Unit.US).time
            - execution_strategy.runtime.to(EventTime.Unit.US).time
            + 1,
            self._time_discretization.time,
        ):
            num_choose_expressions_generated += 1
            # Construct a ChooseExpression for placement at this time.
            # TODO (Sukrit): We just assume for now that all Slots are the same and
            # thus the task can be placed on any Slot. This is not true in general.
            chooseAtTime = tetrisched.strl.ChooseExpression(
                task.unique_name,
                partitions,
                num_slots_required,
                placement_time,
                execution_strategy.runtime.to(EventTime.Unit.US).time,
            )

            # Register this expression with the MAX expression.
            chooseOneFromSet.addChild(chooseAtTime)

        self._logger.debug(
            f"[{current_time.time}] Generated {num_choose_expressions_generated} "
            f"ChooseExpressions for {task.unique_name} from {current_time} to "
            f"{task.deadline-execution_strategy.runtime} for "
            f"{num_slots_required} slots for {execution_strategy.runtime}."
        )
        return chooseOneFromSet

    def _construct_task_graph_strl(
        self,
        current_time: EventTime,
        task: Task,
        task_graph: TaskGraph,
        partitions: tetrisched.Partitions,
        task_strls: Mapping[str, tetrisched.strl.Expression],
        tasks_to_be_scheduled: Optional[List[Task]] = None,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph starting at
        the specified Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The task in the TaskGraph for which the STRL expression is
                to be rooted at.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices for all the Tasks in the TaskGraph and enforces ordering amongst
            them.
        """
        # Check if we have already constructed the STRL for this Task, and return
        # the expression if we have.
        if task.id in task_strls:
            self._logger.debug(
                "[%s] Reusing STRL for Task %s.", current_time.time, task.unique_name
            )
            return task_strls[task.id]

        # Construct the STRL expression for this Task.
        if tasks_to_be_scheduled is not None and task in tasks_to_be_scheduled:
            self._logger.debug(
                f"[{current_time.time}] Constructing the TaskGraph STRL for the "
                f"graph {task_graph.name} rooted at {task.unique_name}."
            )
            task_expression = self.construct_task_strl(current_time, task, partitions)
            task_strls[task.id] = task_expression
        else:
            # If this Task is not in the set of Tasks that we are required to schedule,
            # then we just return a None expression.
            self._logger.debug(
                f"[{current_time.time}] Task {task.unique_name} is not in the set of "
                f"tasks to be scheduled."
            )
            task_expression = None

        # Retrieve the STRL expressions for all the children of this Task.
        child_expressions = []
        for child in task_graph.get_children(task):
            child_expression = self._construct_task_graph_strl(
                current_time,
                child,
                task_graph,
                partitions,
                task_strls,
                tasks_to_be_scheduled,
            )
            if child_expression:
                child_expressions.append(child_expression)

        # If there are no children, return the expression for this Task.
        if len(child_expressions) == 0:
            return task_expression

        # Construct the subtree for the children of this Task.
        if len(child_expressions) > 1:
            # If there are more than one children, then we need to ensure that all
            # of them are placed by collating them under a MinExpression.
            self._logger.debug(
                f"[{current_time.time}] Collating the children of {task.unique_name} "
                f"under a MinExpression {task.unique_name}_children for STRL of the "
                f"TaskGraph {task_graph.name} rooted at {task.unique_name}."
            )
            child_expression = tetrisched.strl.MinExpression(
                f"{task.unique_name}_children"
            )
            for child in child_expressions:
                child_expression.addChild(child)
        else:
            # If there is just one child, then we can just use that subtree.
            child_expression = child_expressions[0]

        # Construct a LessThanExpression to order the two trees.
        # If the current Task has to be scheduled, then we need to ensure that it
        # is scheduled before its children.
        if task_expression:
            self._logger.debug(
                f"[{current_time.time}] Ordering the STRL for {task.unique_name} and "
                f"its children "
                f"{[child.unique_name for child in task_graph.get_children(task)]}"
                f" under a LessThanExpression {task.unique_name}_less_than for "
                f"STRL of the TaskGraph {task_graph.name} rooted at {task.unique_name}."
            )
            task_graph_expression = tetrisched.strl.LessThanExpression(
                f"{task.unique_name}_less_than"
            )
            task_graph_expression.addChild(task_expression)
            task_graph_expression.addChild(child_expression)
        else:
            task_graph_expression = child_expression

        return task_graph_expression

    def construct_task_graph_strl(
        self,
        current_time: EventTime,
        task_graph: TaskGraph,
        partitions: tetrisched.Partitions,
        task_strls: Optional[Mapping[str, tetrisched.strl.Expression]] = None,
        tasks_to_be_scheduled: Optional[List[Task]] = None,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph.

        Args:
            current_time (`EventTime`): The time at which the scheduling is occurring.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.
            partitions (`Partitions`): The partitions that are available for scheduling.
            task_strls (`Mapping[str, tetrisched.strl.Expression]`): A mapping from Task
                IDs to their STRL expressions. Used for caching.
            tasks_to_be_scheduled (`Optional[List[Task]]`): The list of Tasks that are
                to be scheduled. If `None`, then all the Tasks in the TaskGraph are
                considered. Defaults to `None`.
        """
        self._logger.debug(f"The generated STRLs were: {task_strls.keys()}")

        # Construct the STRL expression for all the roots of the TaskGraph.
        root_task_strls = []
        for root in task_graph.get_source_tasks():
            self._logger.debug(
                f"[{current_time.time}] Constructing the STRL for root "
                f"{root.unique_name} while creating the STRL for "
                f"TaskGraph {task_graph.name}."
            )
            root_task_strl = self._construct_task_graph_strl(
                current_time,
                root,
                task_graph,
                partitions,
                task_strls if task_strls else {},
                tasks_to_be_scheduled,
            )
            if root_task_strl:
                root_task_strls.append(root_task_strl)

        if len(root_task_strls) == 0:
            # No roots, possibly empty TaskGraph, return None.
            return None
        elif len(root_task_strls) == 1:
            # Single root, reduce constraints and just bubble this up.
            return root_task_strls[0]
        else:
            # Construct a MinExpression to order the roots of the TaskGraph.
            min_expression_task_graph = tetrisched.strl.MinExpression(
                f"{task_graph.name}_min_expression"
            )
            for root_task_strl in root_task_strls:
                min_expression_task_graph.addChild(root_task_strl)
            return min_expression_task_graph
