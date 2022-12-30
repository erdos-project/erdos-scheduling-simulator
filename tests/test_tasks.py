import pytest

from tests.utils import create_default_task
from utils import EventTime
from workload import BranchPredictionPolicy, Job, Placement, TaskGraph, TaskState


def test_successful_task_creation():
    """Test that a task is created successfully."""
    default_task = create_default_task()
    assert default_task.name == "Perception_Task", "Incorrect name for Task."
    assert default_task.state == TaskState.VIRTUAL, "Incorrect state for Task."


def test_successful_conditional_task_creation():
    """Test that a conditional Task is created successfully."""
    default_task = create_default_task(
        job=Job(
            name="Perception",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        )
    )
    assert default_task.name == "Perception_Task", "Incorrect name for Task."
    assert default_task.conditional, "Incorrect conditionality for Task."


def test_successful_task_release():
    """Test that release() transitions the task to a RELEASED state."""
    default_task = create_default_task()
    release_time = EventTime(2, EventTime.Unit.US)
    default_task.release(release_time)
    assert default_task.release_time == release_time, "Incorrect release time for Task."
    assert default_task.state == TaskState.RELEASED, "Incorrect state for Task."


def test_failed_task_release_without_release_time():
    """Test that a task release without release time fails."""
    default_task = create_default_task()
    with pytest.raises(ValueError):
        default_task.release()


def test_successful_task_release_without_release_time():
    """Test that a task release without release time succeeds."""
    default_task = create_default_task(release_time=2)
    release_time = EventTime(2, EventTime.Unit.US)
    default_task.release()
    assert default_task.release_time == release_time, "Incorrect release time for Task."
    assert default_task.state == TaskState.RELEASED, "Incorrect state for Task."


def test_successful_task_start():
    """Test that a task is successfully started."""
    default_task = create_default_task()
    release_time = EventTime(2, EventTime.Unit.US)
    start_time = EventTime(3, EventTime.Unit.US)
    default_task.release(release_time)
    default_task.schedule(
        release_time,
        Placement(task=default_task, placement_time=release_time, worker_pool_id=None),
    )
    default_task.start(start_time)
    assert default_task.start_time == start_time, "Incorrect start time for Task."
    assert default_task.state == TaskState.RUNNING, "Incorrect state for Task."
    assert default_task._remaining_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect remaining time for Task."


def test_failed_task_start():
    """Test that a task fails to start from an incorrect state."""
    default_task = create_default_task()
    with pytest.raises(ValueError):
        default_task.start(EventTime(3, EventTime.Unit.US))


def test_task_runtime_variability():
    """Test that the runtime of the task can be varied upon its start."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    assert default_task.remaining_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect initial remaining time for the Task."
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            placement_time=EventTime(3, EventTime.Unit.US),
            worker_pool_id=None,
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US), variance=50)
    assert (
        EventTime.zero()
        <= default_task.remaining_time
        < EventTime(2, EventTime.Unit.US)
    ), "Incorrect remaining time for the Task."


def test_successful_task_preempt():
    """Test that a task can be preempted successfully."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task.preempt(EventTime(4, EventTime.Unit.US))
    assert default_task.state == TaskState.PREEMPTED, "Incorrect state for Task."


def test_failed_task_preempt():
    """Test that a Task cannot be PREEMPTED from a non-RUNNING state."""
    default_task = create_default_task()
    with pytest.raises(ValueError):
        default_task.preempt(EventTime(4, EventTime.Unit.US))


def test_successful_task_resume():
    """Test that a task can be resumed successfully."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task.preempt(EventTime(4, EventTime.Unit.US))
    default_task.resume(EventTime(5, EventTime.Unit.US))
    assert default_task.state == TaskState.RUNNING, "Incorrect state for Task."


def test_failed_task_resume():
    """Test that a Task cannot be resumed from a non-preempted state."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    with pytest.raises(ValueError):
        default_task.resume(EventTime(4, EventTime.Unit.US))


def test_task_completion():
    """Test that a Task can be completed successfully."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task._remaining_time = EventTime.zero()
    default_task.finish(EventTime(4, EventTime.Unit.US))
    assert default_task.state == TaskState.COMPLETED, "Incorrect state for Task."


def test_task_eviction():
    """Test that a Task can be evicted successfully."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task.finish(EventTime(4, EventTime.Unit.US))
    assert default_task.state == TaskState.EVICTED, "Incorrect state for Task."


def test_task_step_one():
    """Test that a step() reduces the available time for a Task."""
    default_task = create_default_task(runtime=2)
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    assert default_task._remaining_time == EventTime(
        2, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    default_task.step(EventTime(3, EventTime.Unit.US))
    assert default_task._remaining_time == EventTime(
        1, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    assert default_task._last_step_time == EventTime(
        4, EventTime.Unit.US
    ), "Incorrect last step time in the Task."
    default_task.step(EventTime(4, EventTime.Unit.US))
    assert (
        default_task.remaining_time == EventTime.zero()
    ), "Incorrect remaining time for the Task."
    default_task.finish(EventTime(4, EventTime.Unit.US))
    assert default_task.is_complete(), "Expected the Task to be finished."


def test_task_step_two():
    """Test that a step() works correctly with different step sizes."""
    default_task = create_default_task(runtime=10)
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(5, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(5, EventTime.Unit.US))
    assert default_task.remaining_time == EventTime(
        10, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    default_task.step(EventTime(3, EventTime.Unit.US), EventTime(1, EventTime.Unit.US))
    assert default_task.remaining_time == EventTime(
        10, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    default_task.step(EventTime(4, EventTime.Unit.US), EventTime(2, EventTime.Unit.US))
    assert default_task.remaining_time == EventTime(
        9, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    default_task.step(EventTime(6, EventTime.Unit.US), EventTime(3, EventTime.Unit.US))
    assert default_task.remaining_time == EventTime(
        6, EventTime.Unit.US
    ), "Incorrect remaining time for the Task."
    default_task.step(EventTime(9, EventTime.Unit.US), EventTime(9, EventTime.Unit.US))
    assert (
        default_task.remaining_time == EventTime.zero()
    ), "Incorrect remaining time for the Task."
    default_task.finish(EventTime(9, EventTime.Unit.US))
    assert default_task.is_complete(), "Expected the Task to be finished."


def test_fail_step_non_running():
    """Test that the step() method fails when the task is not running."""
    default_task = create_default_task()
    default_task.release(EventTime(2, EventTime.Unit.US))
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task.preempt(EventTime(4, EventTime.Unit.US))
    assert not default_task.step(
        EventTime(5, EventTime.Unit.US)
    ), "Task should not be completed from a PREEMPTED state."


def test_empty_task_graph_construction():
    """Test that an empty TaskGraph is constructed correctly."""
    task_graph = TaskGraph(name="TestTaskGraph")
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."


def test_task_graph_construction_from_mapping():
    """Test that a TaskGraph is correctly initialized from a Mapping."""
    task_graph = TaskGraph(name="TestTaskGraph", tasks={create_default_task(): []})
    assert len(task_graph) == 1, "Incorrect length of the TaskGraph."


def test_task_addition_to_task_graph():
    """Test addition of Tasks to the graph."""
    default_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    child_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."
    task_graph.add_task(default_task, [child_task])
    assert len(task_graph) == 2, "Incorrect length of the TaskGraph."


def test_addition_of_child_to_task():
    """Test addition of children to a Task."""
    default_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    child_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."
    task_graph.add_task(default_task)
    assert len(task_graph) == 1, "Incorrect length of the TaskGraph."
    task_graph.add_child(default_task, child_task)
    assert len(task_graph) == 2, "Incorrect length of the TaskGraph."


def test_retrieval_of_children():
    """Test that the correct set of children are retrieved."""
    default_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    child_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(default_task, [child_task])
    children = task_graph.get_children(default_task)
    assert len(children) == 1, "Incorrect length of the children retrieved."
    assert id(children[0]) == id(child_task), "Incorrect child returned."


def test_get_schedulable_tasks():
    """Test that the correct set of schedulable tasks are returned."""
    default_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    child_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(default_task, [child_task])
    assert (
        len(task_graph.get_schedulable_tasks(EventTime.zero())) == 0
    ), "Incorrect length of schedulable tasks returned."
    default_task.release(EventTime(2, EventTime.Unit.US))
    assert (
        len(task_graph.get_schedulable_tasks(EventTime(2, EventTime.Unit.US))) == 1
    ), "Incorrect length of schedulable tasks returned."
    child_task.release(EventTime(3, EventTime.Unit.US))
    assert (
        len(task_graph.get_schedulable_tasks(EventTime(3, EventTime.Unit.US))) == 2
    ), "Incorrect length of schedulable tasks returned."
    default_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=default_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    default_task.start(EventTime(3, EventTime.Unit.US))
    default_task.update_remaining_time(EventTime.zero())
    default_task.finish(EventTime(4, EventTime.Unit.US))
    assert (
        len(task_graph.get_schedulable_tasks(EventTime(4, EventTime.Unit.US))) == 1
    ), "Incorrect length of schedulable tasks returned."


def test_release_tasks():
    """Test that the correct tasks are released by the TaskGraph."""
    perception_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    prediction_task = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US))
    )
    planning_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    localization_task = create_default_task(
        job=Job(name="Localization", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(perception_task, [prediction_task])
    task_graph.add_task(prediction_task, [planning_task])
    task_graph.add_task(localization_task)
    assert (
        len(task_graph.get_schedulable_tasks(EventTime.zero())) == 0
    ), "Incorrect length of released tasks returned."

    # Release all available tasks.
    released_tasks = task_graph.get_releasable_tasks()
    assert len(released_tasks) == 2, "Incorrect number of released tasks."
    assert released_tasks == [
        perception_task,
        localization_task,
    ], "Incorrect tasks released by the API."


def test_retrieval_of_parents():
    """Test that the correct set of parents are retrieved."""
    default_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    child_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(default_task, [child_task])
    parents = task_graph.get_parents(child_task)
    assert len(parents) == 1, "Incorrect length of the parents retrieved."
    assert id(parents[0]) == id(default_task), "Incorrect parent returned."


def test_task_completion_notification():
    """Test that the completion of a task ensures release of children."""
    perception_task = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US))
    )
    prediction_task = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US))
    )
    planning_task = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US))
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(perception_task, [planning_task])
    task_graph.add_task(prediction_task, [planning_task])

    released_tasks = task_graph.get_schedulable_tasks(EventTime.zero())
    assert len(released_tasks) == 0, "Incorrect length of released tasks returned."

    # Release and finish the execution of the perception task.
    perception_task.release(EventTime(2, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(EventTime(2, EventTime.Unit.US))
    assert len(released_tasks) == 1, "Incorrect length of released tasks returned."
    assert released_tasks[0] == perception_task, "Incorrect task released."
    perception_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=prediction_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    perception_task.start(EventTime(3, EventTime.Unit.US))
    perception_task.update_remaining_time(EventTime.zero())
    perception_task.finish(EventTime(4, EventTime.Unit.US))

    # Release and finish the execution of the prediction task.
    prediction_task.release(EventTime(4, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(EventTime(4, EventTime.Unit.US))
    assert len(released_tasks) == 1, "Incorrect length of released tasks returned."
    assert released_tasks[0] == prediction_task, "Incorrect tasks released."
    prediction_task.schedule(
        EventTime(4, EventTime.Unit.US),
        Placement(
            task=prediction_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    prediction_task.start(EventTime(4, EventTime.Unit.US))
    prediction_task.update_remaining_time(EventTime.zero())
    prediction_task.finish(EventTime(4, EventTime.Unit.US))
    task_graph.notify_task_completion(prediction_task, EventTime(4, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(EventTime(4, EventTime.Unit.US))
    assert prediction_task.is_complete(), "Task was not completed."
    assert len(released_tasks) == 1, "Incorrect length of released tasks returned."
    assert released_tasks[0] == planning_task, "Incorrect task released."


def test_conditional_task_completion_notification():
    """Test that the completion of a conditional task ensures release of one child."""
    perception_task = create_default_task(
        job=Job(
            name="Perception",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        )
    )
    prediction_task = create_default_task(
        job=Job(
            name="Prediction",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=0.5,
        )
    )
    planning_task = create_default_task(
        job=Job(
            name="Planning", runtime=EventTime(1000, EventTime.Unit.US), probability=0.5
        )
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    release_policy = BranchPredictionPolicy.RANDOM
    task_graph.add_task(perception_task, [planning_task, prediction_task])

    released_tasks = task_graph.get_schedulable_tasks(
        EventTime.zero(), policy=release_policy
    )
    assert len(released_tasks) == 0, "Incorrect length of released tasks returned."

    perception_task.release(EventTime(2, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(
        EventTime(2, EventTime.Unit.US), policy=release_policy
    )
    assert len(released_tasks) == 1, "Incorrect length of released tasks returned."

    # Run and finish the execution of Perception.
    perception_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=perception_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    perception_task.start(EventTime(3, EventTime.Unit.US))
    perception_task.update_remaining_time(EventTime.zero())
    perception_task.finish(EventTime(4, EventTime.Unit.US))
    task_graph.notify_task_completion(perception_task, EventTime(4, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(
        EventTime(4, EventTime.Unit.US), policy=release_policy
    )
    print(released_tasks)
    assert perception_task.is_complete(), "Task was not completed."
    assert (
        len(released_tasks) == 1
    ), "Incorrect number of tasks were released upon completion."
    assert (
        released_tasks[0] == prediction_task or released_tasks[0] == planning_task
    ), "Incorrect task released."


def test_conditional_weighted_task_completion_notification():
    """Test that the completion of a conditional task under a probability distribution
    ensures the release of the correct child."""
    perception_task = create_default_task(
        job=Job(
            name="Perception",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        )
    )
    prediction_task = create_default_task(
        job=Job(
            name="Prediction",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=1.0,
        )
    )
    planning_task = create_default_task(
        job=Job(
            name="Planning", runtime=EventTime(1000, EventTime.Unit.US), probability=0.0
        )
    )
    task_graph = TaskGraph(name="TestTaskGraph")
    task_graph.add_task(perception_task, [planning_task, prediction_task])

    released_tasks = task_graph.get_schedulable_tasks(EventTime.zero())
    assert len(released_tasks) == 0, "Incorrect length of released tasks returned."

    perception_task.release(EventTime(2, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(EventTime(2, EventTime.Unit.US))
    assert len(released_tasks) == 1, "Incorrect length of released tasks returned."

    # Run and finish the execution of Perception.
    perception_task.schedule(
        EventTime(3, EventTime.Unit.US),
        Placement(
            task=perception_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    perception_task.start(EventTime(3, EventTime.Unit.US))
    perception_task.update_remaining_time(EventTime.zero())
    perception_task.finish(EventTime(4, EventTime.Unit.US))
    task_graph.notify_task_completion(perception_task, EventTime(4, EventTime.Unit.US))
    released_tasks = task_graph.get_schedulable_tasks(EventTime(4, EventTime.Unit.US))
    assert perception_task.is_complete(), "Task was not completed."
    assert (
        len(released_tasks) == 1
    ), "Incorrect number of tasks were released upon completion."
    assert released_tasks[0] == prediction_task, "Incorrect task released."


def test_task_graph_index_success():
    """Test that indexing a TaskGraph works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )
    assert len(task_graph) == 6, "Incorrect length of TaskGraph."

    # Slice the TaskGraph.
    task_graph_slice = task_graph[0]
    assert type(task_graph_slice) == TaskGraph, "Incorrect type of slice returned."
    assert len(task_graph_slice) == 3, "Incorrect length of sliced TaskGraph."


def test_task_graph_index_failure():
    """Test that an invalid argument to indexing a TaskGraph fails."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )
    assert len(task_graph) == 6, "Incorrect length of TaskGraph."


def test_task_graph_slice_success():
    """Test that slicing a TaskGraph works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    perception_task_2 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=2,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_2 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=2,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_2 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=2,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1, perception_task_2],
            prediction_task_1: [planning_task_1, prediction_task_2],
            planning_task_1: [planning_task_2],
            perception_task_2: [prediction_task_2],
            prediction_task_2: [planning_task_2],
            planning_task_2: [],
        },
    )
    assert len(task_graph) == 9, "Incorrect length of TaskGraph."

    # Slice the TaskGraph.
    task_graph_slice = task_graph[:2]
    assert type(task_graph_slice) == list, "Incorrect type of slice returned."
    assert len(task_graph_slice) == 2, "Incorrect length of sliced TaskGraph."
    assert len(task_graph_slice[0]) == 3, "Incorrect length of TaskGraph."
    assert len(task_graph_slice[1]) == 3, "Incorrect length of TaskGraph."


def test_is_source_task():
    """Test that the is_source_task method works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that the is_source_task works correctly.
    assert task_graph.is_source_task(
        perception_task_0
    ), "Perception Task is a source task."
    assert task_graph.is_source_task(
        perception_task_1
    ), "Perception Task is a source task."
    assert not task_graph.is_source_task(
        prediction_task_0
    ), "Prediction Task is not a source task."
    assert not task_graph.is_source_task(
        prediction_task_1
    ), "Prediction Task is not a source task."
    assert not task_graph.is_source_task(
        planning_task_0
    ), "Planning Task is not a source task."
    assert not task_graph.is_source_task(
        planning_task_1
    ), "Planning Task is not a source task."


def test_is_sink_task():
    """Test that the is_source_task method works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that the is_sink_task works correctly.
    assert not task_graph.is_sink_task(
        perception_task_0
    ), "Perception Task is not a sink task."
    assert not task_graph.is_sink_task(
        perception_task_1
    ), "Perception Task is not a sink task."
    assert not task_graph.is_sink_task(
        prediction_task_0
    ), "Prediction Task is not a sink task."
    assert not task_graph.is_sink_task(
        prediction_task_1
    ), "Prediction Task is not a sink task."
    assert task_graph.is_sink_task(planning_task_0), "Planning Task is not a sink task."
    assert task_graph.is_sink_task(planning_task_1), "Planning Task is not a sink task."


def test_get_source_tasks():
    """Test that the get_source_tasks method works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that the get_source_tasks works correctly.
    source_tasks = task_graph.get_source_tasks()
    assert len(source_tasks) == 2, "Incorrect length of retrieved source tasks."
    assert set(source_tasks) == {
        perception_task_0,
        perception_task_1,
    }, "Incorrect tasks retrieved by get_source_tasks()"


def test_get_sink_tasks():
    """Test that the get_sink_tasks method works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that the get_source_tasks works correctly.
    sink_tasks = task_graph.get_sink_tasks()
    assert len(sink_tasks) == 2, "Incorrect length of retrieved sink tasks."
    assert set(sink_tasks) == {
        planning_task_0,
        planning_task_1,
    }, "Incorrect tasks retrieved by get_sink_tasks()"


def test_task_graph_complete():
    """Test that the is_complete method works correctly."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_0 = create_default_task(
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_0 = create_default_task(
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0],
            prediction_task_0: [planning_task_0],
            planning_task_0: [],
        },
    )

    # Ensure that the TaskGraph is not complete in the beginning.
    assert not task_graph.is_complete(), "Task Graph should not be complete initially."

    # Mark all tasks as complete.
    perception_task_0.release(EventTime.zero())
    perception_task_0.schedule(
        EventTime.zero(),
        Placement(
            task=perception_task_0, worker_pool_id=None, placement_time=EventTime.zero()
        ),
    )
    perception_task_0.start(EventTime.zero())
    perception_task_0.update_remaining_time(EventTime.zero())
    perception_task_0.finish(EventTime(1, EventTime.Unit.US))
    assert not task_graph.is_complete(), "Task Graph should not be complete."

    released_tasks, _ = task_graph.notify_task_completion(
        perception_task_0, EventTime(1, EventTime.Unit.US)
    )
    assert len(released_tasks) == 1, "Incorrect number of tasks released."
    assert released_tasks[0] == prediction_task_0, "Incorrect task released."
    released_tasks[0].schedule(
        EventTime(1, EventTime.Unit.US),
        Placement(
            task=released_tasks[0],
            worker_pool_id=None,
            placement_time=EventTime(1, EventTime.Unit.US),
        ),
    )
    released_tasks[0].start(EventTime(1, EventTime.Unit.US))
    released_tasks[0].update_remaining_time(EventTime.zero())
    released_tasks[0].finish(EventTime(2, EventTime.Unit.US))
    assert not task_graph.is_complete(), "Task Graph should not be complete."

    released_tasks, _ = task_graph.notify_task_completion(
        prediction_task_0, EventTime(2, EventTime.Unit.US)
    )
    assert len(released_tasks) == 1, "Incorrect number of tasks released."
    assert released_tasks[0] == planning_task_0, "Incorrect task released."
    released_tasks[0].schedule(
        EventTime(2, EventTime.Unit.US),
        Placement(
            task=released_tasks[0],
            worker_pool_id=None,
            placement_time=EventTime(2, EventTime.Unit.US),
        ),
    )
    released_tasks[0].start(EventTime(2, EventTime.Unit.US))
    released_tasks[0].update_remaining_time(EventTime.zero())
    released_tasks[0].finish(EventTime(3, EventTime.Unit.US))
    assert task_graph.is_complete(), "Task Graph should be complete."


def test_conditional_task_graph_complete():
    """Test that the is_complete method works correctly with conditionals."""
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        job=Job(
            name="Perception",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        ),
        timestamp=0,
    )
    prediction_task_0 = create_default_task(
        job=Job(
            name="Prediction",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=0.5,
        ),
        timestamp=0,
    )
    planning_task_0 = create_default_task(
        job=Job(
            name="Planning", runtime=EventTime(1000, EventTime.Unit.US), probability=0.5
        ),
        timestamp=0,
    )
    final_task_0 = create_default_task(
        job=Job("Perception_End", runtime=EventTime.zero(), terminal=True),
        timestamp=0,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, planning_task_0],
            prediction_task_0: [final_task_0],
            planning_task_0: [final_task_0],
        },
    )

    # Ensure that the TaskGraph is not complete in the beginning.
    assert not task_graph.is_complete(), "Task Graph should not be complete initially."

    # Mark all tasks as complete.
    perception_task_0.release(EventTime.zero())
    perception_task_0.schedule(
        EventTime.zero(),
        Placement(
            task=perception_task_0, worker_pool_id=None, placement_time=EventTime.zero()
        ),
    )
    perception_task_0.start(EventTime.zero())
    perception_task_0.update_remaining_time(EventTime.zero())
    perception_task_0.finish(EventTime(1, EventTime.Unit.US))
    assert not task_graph.is_complete(), "Task Graph should not be complete."

    released_tasks, cancelled_tasks = task_graph.notify_task_completion(
        perception_task_0, EventTime(1, EventTime.Unit.US)
    )
    assert len(released_tasks) == 1, "Incorrect number of tasks released."
    if released_tasks[0] == prediction_task_0:
        assert (
            planning_task_0.state == TaskState.CANCELLED
            and planning_task_0 in cancelled_tasks
        ), "The branch not taken was not cancelled."
    elif released_tasks[0] == planning_task_0:
        assert (
            prediction_task_0.state == TaskState.CANCELLED
            and prediction_task_0 in cancelled_tasks
        ), "The branch not taken was not cancelled."
    else:
        assert False, "Incorrect task released."
    released_tasks[0].schedule(
        EventTime(1, EventTime.Unit.US),
        Placement(
            task=released_tasks[0],
            worker_pool_id=None,
            placement_time=EventTime(1, EventTime.Unit.US),
        ),
    )
    released_tasks[0].start(EventTime(1, EventTime.Unit.US))
    released_tasks[0].update_remaining_time(EventTime.zero())
    released_tasks[0].finish(EventTime(2, EventTime.Unit.US))
    assert not task_graph.is_complete(), "Task Graph should not be complete."

    released_tasks, _ = task_graph.notify_task_completion(
        released_tasks[0], EventTime(2, EventTime.Unit.US)
    )
    assert len(released_tasks) == 1, "Incorrect number of tasks released."
    assert released_tasks[0] == final_task_0, "Incorrect task released."
    released_tasks[0].schedule(
        EventTime(2, EventTime.Unit.US),
        Placement(
            task=released_tasks[0],
            worker_pool_id=None,
            placement_time=EventTime(2, EventTime.Unit.US),
        ),
    )
    released_tasks[0].start(EventTime(2, EventTime.Unit.US))
    released_tasks[0].update_remaining_time(EventTime.zero())
    released_tasks[0].finish(EventTime(3, EventTime.Unit.US))
    assert task_graph.is_complete(), "Task Graph should be complete."


def test_task_find():
    # Create the individual tasks.
    perception_task_0 = create_default_task(
        name="Perception_Watermark",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    perception_task_1 = create_default_task(
        name="Perception_Watermark",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    prediction_task_0 = create_default_task(
        name="Prediction_Watermark",
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    prediction_task_1 = create_default_task(
        name="Prediction_Watermark",
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )
    planning_task_0 = create_default_task(
        name="Planning_Watermark",
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    planning_task_1 = create_default_task(
        name="Planning_Watermark",
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that find works correctly.
    perception_tasks = task_graph.find("Perception_Watermark")
    prediction_tasks = task_graph.find("Prediction_Watermark")
    planning_tasks = task_graph.find("Planning_Watermark")
    assert len(perception_tasks) == 2, "Incorrect length of retrieved Perception tasks."
    assert set(perception_tasks) == {
        perception_task_0,
        perception_task_1,
    }, "Incorrect tasks retrieved by the TaskGraph."
    assert len(prediction_tasks) == 2, "Incorrect length of retrieved Perception tasks."
    assert set(prediction_tasks) == {
        prediction_task_0,
        prediction_task_1,
    }, "Incorrect tasks retrieved by the TaskGraph."
    assert len(planning_tasks) == 2, "Incorrect length of retrieved Perception tasks."
    assert set(planning_tasks) == {
        planning_task_0,
        planning_task_1,
    }, "Incorrect tasks retrieved by the TaskGraph."


def test_task_time_dilation():
    # Create the individual tasks.
    localization_task_0 = create_default_task(
        name="Localization_Watermark",
        job=Job(name="Localization", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        release_time=5,
    )
    localization_task_1 = create_default_task(
        name="Localization_Watermark",
        job=Job(name="Localization", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
        release_time=120,
    )
    perception_task_0 = create_default_task(
        name="Perception_Watermark",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        release_time=0,
    )
    perception_task_1 = create_default_task(
        name="Perception_Watermark",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
        release_time=100,
    )
    prediction_task_0 = create_default_task(
        name="Prediction_Watermark",
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        release_time=10,
    )
    prediction_task_1 = create_default_task(
        name="Prediction_Watermark",
        job=Job(name="Prediction", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
        release_time=150,
    )
    planning_task_0 = create_default_task(
        name="Planning_Watermark",
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        release_time=50,
    )
    planning_task_1 = create_default_task(
        name="Planning_Watermark",
        job=Job(name="Planning", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=1,
        release_time=190,
    )

    # Create the TaskGraph.
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            # Timestamp = 0
            localization_task_0: [prediction_task_0, localization_task_1],
            perception_task_0: [prediction_task_0, perception_task_1],
            prediction_task_0: [planning_task_0, prediction_task_1],
            planning_task_0: [planning_task_1],
            # Timestamp = 1
            localization_task_1: [prediction_task_1],
            perception_task_1: [prediction_task_1],
            prediction_task_1: [planning_task_1],
            planning_task_1: [],
        },
    )

    # Check that time dilation works correctly.
    task_graph.dilate(EventTime(50, EventTime.Unit.US))
    assert localization_task_0.release_time == EventTime(
        5, EventTime.Unit.US
    ), "Incorrect release time for Localization task [timestamp=0]"
    assert localization_task_1.release_time == EventTime(
        55, EventTime.Unit.US
    ), "Incorrect release time for Localization task [timestamp=1]"
    assert perception_task_0.release_time == EventTime(
        0, EventTime.Unit.US
    ), "Incorrect release time for Perception task [timestamp=0]"
    assert perception_task_1.release_time == EventTime(
        50, EventTime.Unit.US
    ), "Incorrect release time for Perception task [timestamp=1]"
    assert prediction_task_0.release_time == EventTime(
        10, EventTime.Unit.US
    ), "Incorrect release time for Prediction task [timestamp=0]"
    assert prediction_task_1.release_time == EventTime(
        93, EventTime.Unit.US
    ), "Incorrect release time for Prediction task [timestamp=1]"
    assert planning_task_0.release_time == EventTime(
        50, EventTime.Unit.US
    ), "Incorrect release time for Planning task [timestamp=0]"
    assert planning_task_1.release_time == EventTime(
        133, EventTime.Unit.US
    ), "Incorrect release time for Planning task [timestamp=1]"


def test_task_graph_remaining_time_simple():
    """Checks that the method correctly computes the remaining time
    across a simple TaskGraph."""
    perception_task = create_default_task(
        name="Perception@0",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    prediction_task = create_default_task(
        name="Prediction@0",
        job=Job(name="Prediction", runtime=EventTime(2000, EventTime.Unit.US)),
        runtime=2000,
    )
    planning_task = create_default_task(
        name="Planning@0",
        job=Job(name="Planning", runtime=EventTime(3000, EventTime.Unit.US)),
        runtime=3000,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            perception_task: [prediction_task],
            prediction_task: [planning_task],
            planning_task: [],
        },
    )
    assert task_graph.get_remaining_time() == EventTime(
        6, EventTime.Unit.MS
    ), "Incorrect remaining time returned by Graph."

    # Finish the first task and ensure that the remaining time is correctly computed.
    perception_task.release(EventTime.zero())
    perception_task.schedule(
        EventTime.zero(),
        Placement(
            task=perception_task, worker_pool_id=None, placement_time=EventTime.zero()
        ),
    )
    perception_task.start(EventTime.zero())
    perception_task.update_remaining_time(time=EventTime.zero())
    perception_task.finish(EventTime(1, EventTime.Unit.MS))
    assert task_graph.get_remaining_time() == EventTime(
        5, EventTime.Unit.MS
    ), "Incorrect remaining time returned by the Graph."

    # Finish the second task and ensure that the remaining time is correctly computed.
    prediction_task.release(EventTime(1, EventTime.Unit.MS))
    prediction_task.schedule(
        EventTime(1, EventTime.Unit.MS),
        Placement(
            task=prediction_task,
            worker_pool_id=None,
            placement_time=EventTime(1, EventTime.Unit.US),
        ),
    )
    prediction_task.start(EventTime(1, EventTime.Unit.MS))
    prediction_task.update_remaining_time(time=EventTime.zero())
    prediction_task.finish(EventTime(3, EventTime.Unit.MS))
    assert task_graph.get_remaining_time() == EventTime(
        3, EventTime.Unit.MS
    ), "Incorrect remaining time returned by the Graph."

    # Finish the third task and ensure that the remaining time is correctly computed.
    planning_task.release(EventTime(3, EventTime.Unit.MS))
    planning_task.schedule(
        EventTime(3, EventTime.Unit.MS),
        Placement(
            task=planning_task,
            worker_pool_id=None,
            placement_time=EventTime(3, EventTime.Unit.US),
        ),
    )
    planning_task.start(EventTime(3, EventTime.Unit.MS))
    planning_task.update_remaining_time(time=EventTime.zero())
    planning_task.finish(EventTime(6, EventTime.Unit.MS))
    assert (
        task_graph.get_remaining_time() == EventTime.zero()
    ), "Incorrect remaining time returned by the Graph."
    assert task_graph.is_complete(), "Incorrect completion status returned."


def test_task_graph_remaining_time_complex():
    """Checks that the method correctly computes the remaining time across
    a complex TaskGraph."""
    imu_task = create_default_task(
        name="IMU@0",
        job=Job(name="IMU", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    gnss_task = create_default_task(
        name="GNSS@0",
        job=Job(name="GNSS", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    localization_task = create_default_task(
        name="Localization@0",
        job=Job(name="Localization", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    camera_task = create_default_task(
        name="Camera@0",
        job=Job(name="Camera", runtime=EventTime(5000, EventTime.Unit.US)),
        runtime=5000,
    )
    perception_task = create_default_task(
        name="Perception@0",
        job=Job(name="Perception", runtime=EventTime(1000, EventTime.Unit.US)),
        runtime=1000,
    )
    prediction_task = create_default_task(
        name="Prediction@0",
        job=Job(name="Prediction", runtime=EventTime(2000, EventTime.Unit.US)),
        runtime=2000,
    )
    planning_task = create_default_task(
        name="Planning@0",
        job=Job(name="Planning", runtime=EventTime(3000, EventTime.Unit.US)),
        runtime=3000,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            gnss_task: [localization_task],
            imu_task: [localization_task],
            localization_task: [prediction_task],
            camera_task: [perception_task],
            perception_task: [prediction_task, planning_task],
            prediction_task: [planning_task],
            planning_task: [],
        },
    )
    assert task_graph.get_remaining_time() == EventTime(
        11, EventTime.Unit.MS
    ), "Incorrect remaining time returned by Graph."


def test_task_cancellation():
    """Tests that the Task cancellation process works correctly."""
    detection_start_task = create_default_task(
        name="DetectionStart@0",
        job=Job(
            name="DetectionStart",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        ),
        timestamp=0,
        runtime=1000,
    )
    preprocess_face_task = create_default_task(
        name="PreprocessFace@0",
        job=Job(
            name="PreprocessFace",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=0.5,
        ),
        timestamp=0,
        runtime=1000,
    )
    face_recognition_task = create_default_task(
        name="FaceRecognition",
        job=Job(name="FaceRecognition", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
        runtime=1000,
    )
    preprocess_car_task = create_default_task(
        name="PreprocessCar@0",
        job=Job(
            name="PreprocessCar",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=0.5,
        ),
        timestamp=0,
        runtime=1000,
    )
    car_recognition_task = create_default_task(
        name="CarRecognition@0",
        job=Job(name="CarRecognition", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    detection_end_task = create_default_task(
        name="DetectionEnd@0",
        job=Job(
            name="DetectionEnd",
            runtime=EventTime(1000, EventTime.Unit.US),
            terminal=True,
        ),
        timestamp=0,
        runtime=1000,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            detection_start_task: [preprocess_face_task, preprocess_car_task],
            preprocess_face_task: [face_recognition_task],
            preprocess_car_task: [car_recognition_task],
            face_recognition_task: [detection_end_task],
            car_recognition_task: [detection_end_task],
        },
    )

    # Ensure that the TaskGraph is not complete at the beginning.
    assert not task_graph.is_complete(), "Task Graph should not be complete initially."
    assert all(
        task.state == TaskState.VIRTUAL for task in task_graph.get_nodes()
    ), "All Tasks should be VIRTUAL initially."
    assert (
        len(task_graph.get_schedulable_tasks(EventTime.zero())) == 0
    ), "Incorrect number of currently schedulable tasks."

    # Release and complete the first task.
    detection_start_task.release(EventTime.zero())
    assert (
        len(
            task_graph.get_schedulable_tasks(
                EventTime.zero(), lookahead=EventTime(5000, EventTime.Unit.US)
            )
        )
        == 6
    ), "Incorrect number of currently schedulable tasks."
    detection_start_task.schedule(
        EventTime.zero(),
        Placement(
            task=detection_start_task,
            worker_pool_id=None,
            placement_time=EventTime.zero(),
        ),
    )
    detection_start_task.start(EventTime.zero())
    detection_start_task.update_remaining_time(EventTime.zero())
    detection_start_task.finish(EventTime(1, EventTime.Unit.US))
    assert (
        detection_start_task.state == TaskState.COMPLETED
    ), "DetectionStart@0 should be finished."
    released_tasks, cancelled_tasks = task_graph.notify_task_completion(
        detection_start_task, EventTime(1, EventTime.Unit.US)
    )

    # Check that the correct tasks are released, and the other tasks are correctly
    # cancelled, and their probabilities and remaining time are correctly set.
    assert len(released_tasks) == 1, "Incorrect number of tasks released."
    if released_tasks[0] == preprocess_face_task:
        assert preprocess_face_task.probability == 1.0, "Incorrect probability set."
        assert preprocess_car_task.probability == 0.0, "Incorrect probability set."
        assert (
            preprocess_car_task.state == TaskState.CANCELLED
            and preprocess_car_task in cancelled_tasks
        ), "Incorrect state for Task."
        assert car_recognition_task.probability == 0.0, "Incorrect probability set."
        assert (
            car_recognition_task.state == TaskState.CANCELLED
            and car_recognition_task in cancelled_tasks
        ), "Incorrect state for Task."
        assert (
            detection_end_task.state == TaskState.VIRTUAL
        ), "Incorrect state for Task."
        assert detection_end_task.probability == 1.0, "Incorrect state for Task."
    elif released_tasks[0] == preprocess_car_task:
        assert preprocess_car_task.probability == 1.0, "Incorrect probability set."
        assert preprocess_face_task.probability == 0.0, "Incorrect probability set."
        assert (
            preprocess_face_task.state == TaskState.CANCELLED
            and preprocess_face_task in cancelled_tasks
        ), "Incorrect state for Task."
        assert face_recognition_task.probability == 0.0, "Incorrect probability set."
        assert (
            face_recognition_task.state == TaskState.CANCELLED
            and face_recognition_task in cancelled_tasks
        ), "Incorrect state for Task."
        assert (
            detection_end_task.state == TaskState.VIRTUAL
        ), "Incorrect state for Task."
        assert detection_end_task.probability == 1.0, "Incorrect state for Task."
    else:
        assert False, "Incorrect task released."

    # Check that the set of schedulable tasks is correct.
    schedulable_tasks_now = task_graph.get_schedulable_tasks(
        EventTime(1, EventTime.Unit.US)
    )
    assert (
        len(schedulable_tasks_now) == 1
    ), "Incorrect number of currently schedulable tasks."
    schedulable_tasks_future = task_graph.get_schedulable_tasks(
        EventTime(1, EventTime.Unit.US), lookahead=EventTime(4000, EventTime.Unit.US)
    )
    assert (
        len(schedulable_tasks_future) == 3
    ), "Incorrect number of schedulable tasks in the future."


def test_branch_prediction_policy():
    """Tests that the BranchPredictionPolicy works correctly."""
    detection_start_task = create_default_task(
        name="DetectionStart@0",
        job=Job(
            name="DetectionStart",
            runtime=EventTime(1000, EventTime.Unit.US),
            conditional=True,
        ),
        timestamp=0,
        runtime=1000,
    )
    preprocess_face_task = create_default_task(
        name="PreprocessFace@0",
        job=Job(
            name="PreprocessFace",
            runtime=EventTime(4000, EventTime.Unit.US),
            probability=0.95,
        ),
        timestamp=0,
        runtime=4000,
    )
    face_recognition_task = create_default_task(
        name="FaceRecognition",
        job=Job(name="FaceRecognition", runtime=EventTime(4000, EventTime.Unit.US)),
        timestamp=0,
        runtime=4000,
    )
    preprocess_car_task = create_default_task(
        name="PreprocessCar@0",
        job=Job(
            name="PreprocessCar",
            runtime=EventTime(1000, EventTime.Unit.US),
            probability=0.05,
        ),
        timestamp=0,
        runtime=1000,
    )
    car_recognition_task = create_default_task(
        name="CarRecognition@0",
        job=Job(name="CarRecognition", runtime=EventTime(1000, EventTime.Unit.US)),
        timestamp=0,
    )
    detection_end_task = create_default_task(
        name="DetectionEnd@0",
        job=Job(
            name="DetectionEnd",
            runtime=EventTime(1000, EventTime.Unit.US),
            terminal=True,
        ),
        timestamp=0,
        runtime=1000,
    )
    task_graph = TaskGraph(
        name="TestTaskGraph",
        tasks={
            detection_start_task: [preprocess_face_task, preprocess_car_task],
            preprocess_face_task: [face_recognition_task],
            preprocess_car_task: [car_recognition_task],
            face_recognition_task: [detection_end_task],
            car_recognition_task: [detection_end_task],
        },
    )

    # Test that the WORST_CASE policy works.
    released_tasks = task_graph.resolve_conditional(
        detection_start_task, BranchPredictionPolicy.WORST_CASE
    )
    assert (
        len(released_tasks) == 1
    ), "Incorrect number of tasks released for WORST_CASE prediction policy."
    assert (
        released_tasks[0] == preprocess_car_task
    ), "Incorrect task released for WORST_CASE prediction policy."

    # Test that the BEST_CASE policy works.
    released_tasks = task_graph.resolve_conditional(
        detection_start_task, BranchPredictionPolicy.BEST_CASE
    )
    assert (
        len(released_tasks) == 1
    ), "Incorrect number of tasks released for BEST_CASE prediction policy."
    assert (
        released_tasks[0] == preprocess_face_task
    ), "Incorrect task released for BEST_CASE prediction policy."

    # Test that the MAXIMUM policy works.
    released_tasks = task_graph.resolve_conditional(
        detection_start_task, BranchPredictionPolicy.MAXIMUM
    )
    assert (
        len(released_tasks) == 1
    ), "Incorrect number of tasks released for MAXIMUM prediction policy."
    assert (
        released_tasks[0] == preprocess_face_task
    ), "Incorrect task released for MAXIMUM prediction policy."
