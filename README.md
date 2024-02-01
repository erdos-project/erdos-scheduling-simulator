# ERDOS Simulator

This repository provides a discrete-event simulator for an execution system for DAG-based jobs. The focus of the simulator is to provide an easy playing ground for different scheduling strategies for the execution system. Some features that users can choose to play around with:

- Implement and evaluate DAG and Deadline-Aware scheduling policies that focus on a wide range of commonly used metrics such as maximizing goodput, minimizing average job completion time, minimizing makespan, minimizing placement delay etc.
- Evaluate the effects of real-world perturbations to Task and TaskGraph metrics on the effectiveness of the scheduling strategy. The users can choose to enable the Tasks to randomly perturb their execution time while they are running to test the effectiveness of their constructed schedules.
- Implement and evaluate heterogeneous and multi-dimensional resource-focused scheduling strategies. The simulator provides the ability to define a cluster with different types and instances of resources, and provide various execution strategies for the Tasks such that a preference order over their choices can be enforced by the scheduler.

## Installation

To get the repository up and running, install Python 3.7+ and set up a virtual environment. Inside the virtual environment,
run

```console
pip install -r requirements.txt
```

If you aim to use the C++-based scheduler defined in `schedulers/tetrisched`, refer to its README for installation instructions and ensure that the package is available in the correct virtual environment.

## Terminology

- *Job*: A Job is a static instance of a piece of computation that is not invokable, and is used to capture some metadata about its execution. For example, a Job captures the resource requirements of the computation, and its expected runtime.
- *JobGraph*: A JobGraph is a static entity that captures the relations amongst Jobs, such that if there is an edge from Job A to Job B, then Job A must finish before Job B can execute.
- *Task*: A dynamic instantiation of the static *Job*, a Task provides additional runtime information such as its deadline, its release time, the time at which it started executing, the time at which it finished execution and the state that the Task is currently in.
- *TaskGraph*: A dynamic instantiation of the static *JobGraph*, a TaskGraph provides runtime information such as the progress of the tasks in the TaskGraph, the end-to-end deadline to which the TaskGraph is being enforced, and other helper functions that can be used to easily query information about the particular invocation of the TaskGraph.
- *ExecutionStrategy*: An ExecutionStrategy object defines the resource requirements and runtime of a particular viable strategy for the Task to execute with. The strategy defines that if the Task is provided with the given resource requirements, then it will run within the given runtime.
- *WorkProfile*: Each Task in a TaskGraph is associated with a *WorkProfile* that summarizes all the different strategies with which a Task can execute.
- *Workload*: A Workload object provides the simulator with the information about the Tasks and TaskGraphs that are to be executed in this given run. Users must define specific Workloads, or implement their own data loaders that can generate the Workload object.
- *Worker*: A Worker is a collection of (possibly) heterogeneous resources that forms the boundary of scheduling. Unless specified, a Task cannot use resources from multiple Workers concurrently.
- *WorkerPool*: A collection of *Worker*s that is used to represent the state of the cluster to the scheduler.

## Basic Scheduling Example

An easy way to get started is to define a simple Workload and a WorkerPool and pass it to the simulator to execute under a specific scheduling policy. For our purposes, we specify a simplistic version of the Autonomous Vehicle (AV) pipeline shown in Gog et al. ([EuroSys '22](https://dl.acm.org/doi/pdf/10.1145/3492321.3519576)). Simple workloads are defined in a JSON / YAML specification, and the representation of the AV pipeline can be seen in [simple_av_workload.yaml](./profiles/workload/simple_av_workload.yaml). The set of TaskGraphs to be released in the system is defined using the graphs parameter as shown below:

```yaml
graphs:
    - name: AutonomousVehicle
      graph: ...
      release_policy: fixed
      period: 100 # In microseconds.
      invocations: 10
      deadline_variance: [50, 100]
```

The above example defines a graph named "AutonomousVehicle" that is released at a fixed interval of 100 microseconds for 10 invocations, each of which is randomly assigned a deadline slack equivalent to somewhere between 50 and a 100% of the runtime of the critical path of the pipeline.

The graph definition is given as a series of nodes, each of which defines the WorkProfile with which it can run, and the set of children that become available once it finishes its execution. For example, the `Perception` task below can be run with the `PerceptionProfile` by using 1 GPU and finishing within 200 microseconds. Once the `Perception` task is finished, the `Prediction` task becomes available for execution.

```yaml
      graph: ...
          - name: Perception
            work_profile: PerceptionProfile
            children: ["Prediction"]
      profiles: ...
            - name: PerceptionProfile
            execution_strategies:
                - batch_size: 1
                    runtime: 200
                    resource_requirements:
                        GPU:any: 1
```

Similarly, a WorkerPool is defined as a collection of resources available in the cluster, and an example can be seen in [worker_1_machine_1_gpu_profile.yaml](./profiles/workers/worker_1_machine_1_gpu_profile.yaml), and defines a WorkerPool with 1 GPU as follows:

```yaml
- name: WorkerPool_1
  workers:
      - name: Worker_1_1
        resources:
            - name: GPU
              quantity: 1
```

## Running the Example

The easiest way to run an example is to define a configuration file for the flag values to main.py. For running the above example with an AV pipeline, a sample configuration has been provided in [simple_av_workload.conf](./configs/simple_av_workload.conf), which defines the names of the log and the CSV files, along with the scheduler that is to be used to place tasks on the WorkerPool.

To run this example, simply run

```bash
python main.py --flagfile=configs/simple_av_workload.conf
```


## Questions / Comments?

Please feel free to raise issues / PRs for bugs that you encounter or enhancements that you would like to see!
