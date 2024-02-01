# Schedulers

The package contains implementations for various scheduling policies that are supported by the simulator out-of-the-box.

The simulator can be configured to use any of the available scheduling policies through the flag `--scheduler`. The simulator supports various TaskGraph-unaware scheduling policies such as:

- `EDF` for an Earliest-Deadline First scheduler that considers all the available tasks for scheduling and matches them to the available resources.
- `FIFO` is a First-Come First-Serve scheduler that orders the available tasks based on their release times, and greedily matches them to the available resources.
- `LSF` is a Least-Slack first scheduler that orders the available tasks by the remaining slack between the current time and their deadline, and greedily matches them to the available resources.
- `TetriSched_[CPLEX,Gurobi]` is a re-implementation of the
TetriSched scheduler from the [EuroSys '16 paper](https://dl.acm.org/doi/pdf/10.1145/2901318.2901355) that implements
plan-ahead, packing-aware scheduling for tasks while respecting their deadlines.

In addition, the simulator also has some TaskGraph-aware scheduling policies, defined in the following schedulers:

- `ILP` is a mathematical programming based scheduler that implements the TaskGraph scheduling as an optimization program with quadratic constraints. Note that, while this scheduler can provide extremely accurate placement decisions, it cannot scale beyond simple examples.
- `TetriSched` is a C++-based mathematical optimization-backed scheduler that formulates the TaskGraph scheduling problem declaratively, and abstracts away various modeling decisions that lead to faster performance.

## Configuring the Schedulers.
The simulator provides various relevant flags that can be used to modulate the behavior of different schedulers and explore accuracy-latency tradeoffs on a given workload.

| Flag            | Description |
| -----------     | ----------- |
| preemption      | This flag can be set to decide if currently running tasks are allowed to be preempted to be able to schedule other available tasks.       |
| enforce_deadlines       | This flag is used to decide how to handle tasks that are past their deadline. If set, the schedulers will enforce the deadlines and drop tasks that cannot meet their deadline.        |
| scheduler_runtime | A flag to modulate the runtime of the scheduler. If set, the scheduler informs the simulator that it finished in the given time. |
| release_taskgraphs | If True, the TaskGraphs are made available to the scheduler. By default, only the runnable tasks are seen by the schedulers. |
| scheduler_plan_ahead | For schedulers that plan the placements of the tasks in the future, this flag dictates how long the planning horizon is. |
| scheduler_look_ahead | A flag that provides the schedulers with the ability to see incoming tasks between the `scheduler_look_ahead` time units away from the current time. |
| retract_schedules | For schedulers that plan ahead, setting this to True allows the schedulers to change the prior placements of the tasks. However, this does not change the placement of already running tasks. To enable that, use `--preemption` |
| goal | For mathematical programming-based schedulers, the user can choose between a `max_goodput` and a `min_placement_delay` goal for the solver. |



## Defining Schedulers.

Every scheduler must adhere to the interface defined in `base_scheduler.py`. Specifically, the scheduler must implement the following method:

```python
def schedule(
        self,
        sim_time: EventTime,
        workload: Workload,
        worker_pools: "WorkerPools",
    ) -> Placements:
```

This method takes the current time, the `Workload` object that provides information about the currently available tasks and taskgraphs for scheduling, and the `WorkerPools` object that provides information about the resources available in the system. The scheduler must then generate a series of `Placement` decisions that map each `Task` to a given `Worker` or a `WorkerPool`.
