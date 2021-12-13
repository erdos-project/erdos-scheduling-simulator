import pytest

from workload import Job, Resource, Resources, TaskState, Task, TaskGraph


def __create_default_task(
        job=Job(name="Perception"),
        resource_requirements=Resources(
                        resource_vector={Resource(name="CPU", _id="any"): 1}),
        runtime=1.0,
        deadline=10.0,
        timestamp=0,
        release_time=-1,
        start_time=-1,
        completion_time=-1,
):
    """ Helper function to create a default task. """
    return Task(name="{}_Task".format(job.name),
                job=job,
                resource_requirements=resource_requirements,
                runtime=runtime,
                deadline=deadline,
                timestamp=timestamp,
                release_time=release_time,
                start_time=start_time,
                completion_time=completion_time)


def test_successful_task_creation():
    """ Test that a task is created successfully. """
    default_task = __create_default_task()
    assert default_task.name == "Perception_Task", "Incorrect name for Task."
    assert default_task.state == TaskState.VIRTUAL, "Incorrect state for Task."


def test_successful_task_release():
    """ Test that release() transitions the task to a RELEASED state. """
    default_task = __create_default_task()
    default_task.release(2.0)
    assert default_task.release_time == 2.0, "Incorrect release time for Task."
    assert default_task.state == TaskState.RELEASED,\
        "Incorrect state for Task."


def test_failed_task_release_without_release_time():
    """ Test that a task release without release time fails. """
    default_task = __create_default_task()
    with pytest.raises(ValueError):
        default_task.release()


def test_successful_task_release_without_release_time():
    """ Test that a task release without release time succeeds. """
    default_task = __create_default_task(release_time=2.0)
    default_task.release()
    assert default_task.release_time == 2.0, "Incorrect release time for Task."
    assert default_task.state == TaskState.RELEASED,\
        "Incorrect state for Task."


def test_successful_task_start():
    """ Test that a task is successfully started. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    assert default_task.start_time == 3.0, "Incorrect start time for Task."
    assert default_task.state == TaskState.RUNNING, "Incorrect state for Task."
    assert default_task._remaining_time == 1.0,\
        "Incorrect remaining time for Task."


def test_failed_task_start():
    """ Test that a task fails to start from an incorrect state."""
    default_task = __create_default_task()
    with pytest.raises(ValueError):
        default_task.start(3.0)


def test_task_runtime_variability():
    """ Test that the runtime of the task can be varied upon its start. """
    default_task = __create_default_task()
    default_task.release(2.0)
    assert default_task.remaining_time == 1.0,\
        "Incorrect initial remaining time for the Task."
    default_task.start(3.0, variance=50.0)
    assert 0 <= default_task.remaining_time <= 1.50,\
        "Incorrect remaining time for the Task."


def test_successful_task_pause():
    """ Test that a task can be paused successfully. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    default_task.pause(4.0)
    assert default_task.state == TaskState.PAUSED, "Incorrect state for Task."


def test_failed_task_pause():
    """ Test that a Task cannot be PAUSED from a non-RUNNING state. """
    default_task = __create_default_task()
    with pytest.raises(ValueError):
        default_task.pause(4.0)


def test_successful_task_resume():
    """ Test that a task can be resumed successfully. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    default_task.pause(4.0)
    default_task.resume(5.0)
    assert default_task.state == TaskState.RUNNING, "Incorrect state for Task."


def test_failed_task_resume():
    """ Test that a Task cannot be resumed from a non-paused state. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    with pytest.raises(ValueError):
        default_task.resume(4.0)


def test_task_completion():
    """ Test that a Task can be completed successfully. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    default_task._remaining_time = 0
    default_task.finish(4.0)
    assert default_task.state == TaskState.COMPLETED,\
        "Incorrect state for Task."


def test_task_eviction():
    """ Test that a Task can be evicted successfully. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    default_task.finish(4.0)
    assert default_task.state == TaskState.EVICTED,\
        "Incorrect state for Task."


def test_task_step_one():
    """ Test that a step() reduces the available time for a Task. """
    default_task = __create_default_task(runtime=2.0)
    default_task.release(2.0)
    default_task.start(3.0)
    assert default_task._remaining_time == 2.0,\
        "Incorrect remaining time for the Task."
    default_task.step(3.0)
    assert default_task._remaining_time == 1.0,\
        "Incorrect remaining time for the Task."
    assert default_task._last_step_time == 4.0,\
        "Incorrect last step time in the Task."
    default_task.step(4.0)
    assert default_task._remaining_time == 0.0,\
        "Incorrect remaining time for the Task."
    assert default_task.is_complete(), "Expected the Task to be finished."


def test_task_step_two():
    """ Test that a step() works correctly with different step sizes. """
    default_task = __create_default_task(runtime=10.0)
    default_task.release(2.0)
    default_task.start(5.0)
    assert default_task._remaining_time == 10.0,\
        "Incorrect remaining time for the Task."
    default_task.step(3.0, 1.0)
    assert default_task._remaining_time == 10.0,\
        "Incorrect remaining time for the Task."
    default_task.step(4.0, 2.0)
    assert default_task._remaining_time == 9.0,\
        "Incorrect remaining time for the Task."
    default_task.step(6.0, 3.0)
    assert default_task._remaining_time == 6.0,\
        "Incorrect remaining time for the Task."
    default_task.step(9.0, 9.0)
    assert default_task._remaining_time == 0.0,\
        "Incorrect remaining time for the Task."
    assert default_task.is_complete(), "Expected the Task to be finished."


def test_fail_step_non_running():
    """ Test that the step() method fails when the task is not running. """
    default_task = __create_default_task()
    default_task.release(2.0)
    default_task.start(3.0)
    default_task.pause(4.0)
    assert not default_task.step(5.0),\
        "Task should not be completed from a PAUSED state."


def test_empty_task_graph_construction():
    """ Test that an empty TaskGraph is constructed correctly. """
    task_graph = TaskGraph()
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."


def test_task_graph_construction_from_mapping():
    """ Test that a TaskGraph is correctly initialized from a Mapping. """
    task_graph = TaskGraph(tasks={__create_default_task(): []})
    assert len(task_graph) == 1, "Incorrect length of the TaskGraph."


def test_task_addition_to_task_graph():
    """ Test addition of Tasks to the graph. """
    default_task = __create_default_task(job=Job(name="Perception"))
    child_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."
    task_graph.add_task(default_task, [child_task])
    assert len(task_graph) == 2, "Incorrect length of the TaskGraph."


def test_addition_of_child_to_task():
    """ Test addition of children to a Task. """
    default_task = __create_default_task(job=Job(name="Perception"))
    child_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    assert len(task_graph) == 0, "Incorrect length of the TaskGraph."
    task_graph.add_task(default_task)
    assert len(task_graph) == 1, "Incorrect length of the TaskGraph."
    task_graph.add_child(default_task, child_task)
    assert len(task_graph) == 2, "Incorrect length of the TaskGraph."


def test_retrieval_of_children():
    """ Test that the correct set of children are retrieved. """
    default_task = __create_default_task(job=Job(name="Perception"))
    child_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    task_graph.add_task(default_task, [child_task])
    children = task_graph.get_children(default_task)
    assert len(children) == 1, "Incorrect length of the children retrieved."
    assert id(children[0]) == id(child_task), "Incorrect child returned."


def test_get_released_tasks():
    """ Test that the correct set of released tasks are returned. """
    default_task = __create_default_task(job=Job(name="Perception"))
    child_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    task_graph.add_task(default_task, [child_task])
    assert len(task_graph.get_released_tasks()) == 0,\
        "Incorrect length of released tasks returned."
    default_task.release(2.0)
    assert len(task_graph.get_released_tasks()) == 1,\
        "Incorrect length of released tasks returned."
    child_task.release(3.0)
    assert len(task_graph.get_released_tasks()) == 2,\
        "Incorrect length of released tasks returned."
    default_task.finish(4.0)
    assert len(task_graph.get_released_tasks()) == 1,\
        "Incorrect length of released tasks returned."


def test_release_tasks():
    """ Test that the correct tasks are released by the TaskGraph. """
    perception_task = __create_default_task(job=Job(name="Perception"))
    prediction_task = __create_default_task(job=Job(name="Prediction"))
    planning_task = __create_default_task(job=Job(name="Planning"))
    localization_task = __create_default_task(job=Job(name="Localization"))
    task_graph = TaskGraph()
    task_graph.add_task(perception_task, [prediction_task])
    task_graph.add_task(prediction_task, [planning_task])
    task_graph.add_task(localization_task)
    assert len(task_graph.get_released_tasks()) == 0,\
        "Incorrect length of released tasks returned."

    # Release all available tasks.
    released_tasks = task_graph.release_tasks(1.0)
    assert len(released_tasks) == 2, "Incorrect number of released tasks."
    assert released_tasks == [perception_task, localization_task],\
        "Incorrect tasks released by the API."


def test_retrieval_of_parents():
    """ Test that the correct set of parents are retrieved. """
    default_task = __create_default_task(job=Job(name="Perception"))
    child_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    task_graph.add_task(default_task, [child_task])
    parents = task_graph.get_parents(child_task)
    assert len(parents) == 1, "Incorrect length of the parents retrieved."
    assert id(parents[0]) == id(default_task), "Incorrect parent returned."


def test_task_completion_notification():
    """ Test that the completion of a task ensures release of children. """
    perception_task = __create_default_task(job=Job(name="Perception"))
    prediction_task = __create_default_task(job=Job(name="Prediction"))
    planning_task = __create_default_task(job=Job(name="Planning"))
    task_graph = TaskGraph()
    task_graph.add_task(perception_task, [planning_task])
    task_graph.add_task(prediction_task, [planning_task])

    released_tasks = task_graph.get_released_tasks()
    assert len(released_tasks) == 0,\
        "Incorrect length of released tasks returned."

    perception_task.release(2.0)
    released_tasks = task_graph.get_released_tasks()
    assert len(released_tasks) == 1,\
        "Incorrect length of released tasks returned."
    assert released_tasks[0] == perception_task,\
        "Incorrect task released."

    prediction_task.release(2.0)
    released_tasks = task_graph.get_released_tasks()
    assert len(released_tasks) == 2,\
        "Incorrect length of released tasks returned."
    assert released_tasks[0] == perception_task,\
        "Incorrect task released."
    assert released_tasks[1] == prediction_task,\
        "Incorrect task released."

    # Run and finish the execution of Perception.
    perception_task.start(3.0)
    perception_task.update_remaining_time(0)
    perception_task.finish(4.0)
    task_graph.notify_task_completion(perception_task, 4.0)
    released_tasks = task_graph.get_released_tasks()
    assert perception_task.is_complete(), "Task was not completed."
    assert len(released_tasks) == 1,\
        "Incorrect length of released tasks returned."
    assert released_tasks[0] == prediction_task,\
        "Incorrect task released."

    # Run and finish the execution of Prediction.
    prediction_task.start(3.0)
    prediction_task.update_remaining_time(0)
    prediction_task.finish(4.0)
    task_graph.notify_task_completion(prediction_task, 4.0)
    released_tasks = task_graph.get_released_tasks()
    assert prediction_task.is_complete(), "Task was not completed."
    assert len(released_tasks) == 1,\
        "Incorrect length of released tasks returned."
    assert released_tasks[0] == planning_task,\
        "Incorrect task released."
