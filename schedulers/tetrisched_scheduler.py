import time
from typing import List, Optional

import absl  # noqa: F401
import tetrisched_py as tetrisched

from schedulers import BaseScheduler
from utils import EventTime
from workers import WorkerPools
from workload import Placements, Resource, Task, TaskGraph, Workload


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
        if len(tasks_to_be_scheduled) > 0:
            # Construct the partitions from the Workers in the WorkerPool.
            partitions = self.construct_partitions(worker_pools=worker_pools)
            for task in tasks_to_be_scheduled:
                # Construct the STRL expression for the task.
                self.construct_task_strl(
                    current_time=sim_time, task=task, partitions=partitions
                )
            raise NotImplementedError("TetrischedScheduler is not implemented yet.")
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

                # Create a tetrisched Worker.
                tetrisched_worker = tetrisched.Worker(worker_index, worker.name)
                slot_quantity = worker.resources.get_total_quantity(
                    resource=Resource(name="Slot", _id="any")
                )
                partition = tetrisched.Partition()
                partition.addWorker(tetrisched_worker, slot_quantity)
                partitions.addPartition(partition)
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
            f"{task.name} with name: {task.unique_name}_placement."
        )
        chooseOneFromSet = tetrisched.strl.MaxExpression(
            f"{task.unique_name}_placement"
        )

        # Construct the STRL ChooseExpressions for this Task.
        # This expression represents a particular placement choice for this Task.
        num_slots_required = execution_strategy.resources.get_total_quantity(
            resource=Resource(name="Slot", _id="any")
        )

        for placement_time in range(
            current_time.to(EventTime.Unit.US).time,
            task.deadline.to(EventTime.Unit.US).time,
            self._time_discretization.time,
        ):
            self._logger.debug(
                f"[{current_time.time}] Generating a Choose Expression "
                f"for {task.unique_name} at time {placement_time} for "
                f"{num_slots_required} slots for {execution_strategy.runtime}."
            )
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
        return chooseOneFromSet

    def construct_task_graph_strl(
        self,
        current_time: EventTime,
        task: Task,
        task_graph: TaskGraph,
        partitions: tetrisched.Partitions,
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
        # Construct the STRL expression for this Task.
        self._logger.debug(
            f"[{current_time.time}] Constructing the TaskGraph STRL for the "
            f"graph {task_graph.name} rooted at {task.unique_name}."
        )
        task_expression = self.construct_task_strl(current_time, task, partitions)

        # Retrieve the STRL expressions for all the children of this Task.
        child_expressions = []
        for child in task_graph.get_children(task):
            self._logger.debug(
                f"[{current_time.time}] Constructing the STRL for {child.unique_name} "
                f"while creating the STRL for TaskGraph {task_graph.name} rooted at "
                f"{task.unique_name}."
            )
            child_expressions.append(
                self.construct_task_graph_strl(
                    current_time, child, task_graph, partitions
                )
            )

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
        self._logger.debug(
            f"[{current_time.time}] Ordering the STRL for {task.unique_name} and its "
            f"children under a LessThanExpression {task.unique_name}_less_than for "
            f"STRL of the TaskGraph {task_graph.name} rooted at {task.unique_name}."
        )
        task_graph_expression = tetrisched.strl.LessThanExpression(
            f"{task.unique_name}_less_than"
        )
        task_graph_expression.addChild(task_expression)
        task_graph_expression.addChild(child_expression)

        return task_graph_expression
