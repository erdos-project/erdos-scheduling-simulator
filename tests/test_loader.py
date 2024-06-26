from data import TaskLoader, TaskLoaderPylot, WorkerLoader, WorkloadLoader
from tests.test_tasks import create_default_task
from utils import EventTime
from workload import (
    ExecutionStrategies,
    ExecutionStrategy,
    Job,
    JobGraph,
    Resource,
    Resources,
    WorkProfile,
)


def test_create_jobs():
    """Tests the __create_jobs method of the TaskLoaderPylot."""
    perception_job = Job(
        name="perception_operator",
        profile=WorkProfile(
            name="perception_operator_work_profile",
            execution_strategies=ExecutionStrategies(
                [
                    ExecutionStrategy(
                        resources=Resources(), batch_size=1, runtime=EventTime.zero()
                    )
                ]
            ),
        ),
    )
    prediction_job = Job(
        name="prediction_operator",
        profile=WorkProfile(
            name="prediction_operator_work_profile",
            execution_strategies=ExecutionStrategies(
                [
                    ExecutionStrategy(
                        resources=Resources(), batch_size=1, runtime=EventTime.zero()
                    )
                ]
            ),
        ),
    )
    planning_job = Job(
        name="planning_operator",
        profile=WorkProfile(
            name="planning_operator_work_profile",
            execution_strategies=ExecutionStrategies(
                [
                    ExecutionStrategy(
                        resources=Resources(), batch_size=1, runtime=EventTime.zero()
                    )
                ]
            ),
        ),
    )
    jobs = TaskLoaderPylot._TaskLoaderPylot__create_jobs(
        [
            {"pid": "perception_operator", "dur": 1000},
            {"pid": "prediction_operator", "dur": 1000},
            {"pid": "planning_operator", "dur": 1000},
        ],
        JobGraph(
            name="TestJobGraph",
            jobs={
                perception_job: [prediction_job],
                prediction_job: [planning_job],
                planning_job: [],
            },
        ),
    )
    assert len(jobs) == 3, "Incorrect number of Jobs returned."
    assert jobs["perception_operator"].name == "perception_operator" and jobs[
        "perception_operator"
    ].execution_strategies.get_fastest_strategy().runtime == EventTime(
        1000, EventTime.Unit.US
    ), "Incorrect Job returned."


def test_create_resources():
    """Tests the __create_resources method of the WorkloadLoader."""

    resources_json = [
        {
            "name": "perception_operator",
            "resource_requirements": {"CPU:any": 1},
        },
        {
            "name": "prediction_operator",
            "resource_requirements": {"CPU:any": 1, "GPUTaskLoaderPylot:any": 1},
        },
        {
            "name": "planning_operator",
            "resource_requirements": {"CPU:any": 1, "GPU:any": 1},
        },
    ]
    resources = {}
    for operator in resources_json:
        resources[operator["name"]] = WorkloadLoader._WorkloadLoader__create_resources(
            operator["resource_requirements"]
        )

    assert len(resources) == 3, "Incorrect number of Resources returned."
    assert (
        len(resources["perception_operator"]) == 1
    ), "Incorrect number of Resources returned."
    assert (
        len(resources["prediction_operator"]) == 2
    ), "Incorrect number of Resources returned."
    assert (
        len(resources["planning_operator"]) == 2
    ), "Incorrect number of Resources returned."


def test_create_tasks():
    """Tests the __create_tasks method of the TaskLoaderPylot."""
    json_entries = [
        {
            "name": "perception_operator.on_watermark",
            "pid": "perception_operator",
            "args": {
                "timestamp": 1,
            },
            "dur": 100,
            "ts": 50,
        }
    ]
    jobs = {
        "perception_operator": Job(
            name="Perception",
            profile=WorkProfile(
                name="Perception_Work_Profile",
                execution_strategies=ExecutionStrategies(
                    [
                        ExecutionStrategy(
                            resources=Resources(
                                resource_vector={Resource(name="CPU", _id="any"): 1}
                            ),
                            batch_size=1,
                            runtime=EventTime(1000, EventTime.Unit.US),
                        )
                    ]
                ),
            ),
        ),
    }
    tasks = TaskLoaderPylot._TaskLoaderPylot__create_tasks(
        json_entries,
        "test_task_graph",
        jobs,
    )

    assert len(tasks) == 1, "Incorrect number of Tasks returned."
    assert (
        tasks[0].name == "perception_operator.on_watermark"
    ), "Incorrect name returned for the Task."
    assert tasks[
        0
    ].available_execution_strategies.get_fastest_strategy().runtime == EventTime(
        100, EventTime.Unit.US
    ), "Incorrect runtime returned for the Task."
    assert tasks[0].timestamp == 1, "Incorrect timestamp for the Task."
    assert (
        jobs["perception_operator"] == tasks[0].job
    ), "Incorrect Job returned for the Task."


def test_create_jobgraph():
    """Tests the construction of a JobGraph by the TaskLoaderPylot."""
    jobs = {
        "perception_operator": Job(name="Perception"),
        "prediction_operator": Job(name="Prediction"),
        "planning_operator": Job(name="Planning"),
    }
    edges = [
        ("perception_operator", "prediction_operator"),
        ("perception_operator", "planning_operator"),
        ("prediction_operator", "planning_operator"),
    ]
    job_graph = TaskLoaderPylot._TaskLoaderPylot__create_job_graph(
        jobs, "test_job_graph", edges
    )

    assert len(job_graph) == 3, "Incorrect length for JobGraph."
    assert (
        len(job_graph.get_children(jobs["perception_operator"])) == 2
    ), "Incorrect length of children for Job."
    assert (
        len(job_graph.get_children(jobs["prediction_operator"])) == 1
    ), "Incorrect length of children for Job."
    assert (
        len(job_graph.get_children(jobs["planning_operator"])) == 0
    ), "Incorrect length of children for Job."


def test_create_taskgraph():
    """Tests the construction of a TaskGraph by the TaskLoaderPylot."""
    # Create the JobGraph first.
    jobs = {
        "perception_operator": Job(
            name="Perception",
            pipelined=True,
        ),
        "prediction_operator": Job(name="Prediction"),
        "planning_operator": Job(name="Planning"),
    }
    edges = [
        ("perception_operator", "prediction_operator"),
        ("perception_operator", "planning_operator"),
        ("prediction_operator", "planning_operator"),
    ]
    job_graph = TaskLoaderPylot._TaskLoaderPylot__create_job_graph(
        jobs, "test_job_graph", edges
    )

    # Create a list of Tasks to be put into a graph.
    tasks = [
        create_default_task(job=jobs["perception_operator"], timestamp=1),
        create_default_task(job=jobs["perception_operator"], timestamp=2),
        create_default_task(job=jobs["perception_operator"], timestamp=3),
        create_default_task(job=jobs["prediction_operator"], timestamp=1),
        create_default_task(job=jobs["prediction_operator"], timestamp=2),
        create_default_task(job=jobs["prediction_operator"], timestamp=3),
        create_default_task(job=jobs["planning_operator"], timestamp=1),
        create_default_task(job=jobs["planning_operator"], timestamp=2),
        create_default_task(job=jobs["planning_operator"], timestamp=3),
    ]

    # Create a TaskGraph using the jobs and the list of tasks.
    _, task_graph = TaskLoader._TaskLoader__create_task_graph(
        "TestTaskGraph", tasks, job_graph
    )
    assert len(task_graph) == len(tasks), "Incorrect length of TaskGraph."

    # Check the parent-child relationships.
    parents_perception_task_1 = task_graph.get_parents(tasks[0])
    assert len(parents_perception_task_1) == 0, "Incorrect length for parents."

    parents_perception_task_3 = task_graph.get_parents(tasks[2])
    assert len(parents_perception_task_3) == 0, "Incorrect length for parents."

    parents_prediction_task = task_graph.get_parents(tasks[5])
    assert len(parents_prediction_task) == 2, "Incorrect length for parents."
    assert set(parents_prediction_task) == {
        tasks[2],
        tasks[4],
    }, "Incorrect parents for the task."

    parents_planning_task = task_graph.get_parents(tasks[8])
    assert len(parents_planning_task) == 3, "Incorrect length for parents."
    assert set(parents_planning_task) == {
        tasks[2],
        tasks[5],
        tasks[7],
    }, "Incorrect parents for the task."


def test_create_worker_pool():
    """Tests the construction of the WorkerPool by the WorkerLoaderJSON."""
    # Create a test WorkerPool topology.
    worker_pools = WorkerLoader._WorkerLoader__create_worker_pools(
        [
            {
                "name": "WorkerPool_1_1",
                "workers": [
                    {
                        "name": "Worker_1_1",
                        "resources": [{"name": "CPU", "quantity": 5}],
                    }
                ],
            }
        ],
        scheduler=None,
    )

    assert len(worker_pools) == 1, "Incorrect number of WorkerPools returned."
    assert (
        worker_pools[0].name == "WorkerPool_1_1"
    ), "Incorrect name for the WorkerPool."
    assert len(worker_pools[0]) == 1, "Incorrect number of Workers in WorkerPool."
    assert (
        worker_pools[0].workers[0].name == "Worker_1_1"
    ), "Incorrect name for the Worker."
    assert (
        worker_pools[0]
        .workers[0]
        .resources.get_available_quantity(Resource(name="CPU", _id="any"))
        == 5
    ), "Incorrect quantity of CPU resources in the Worker."
