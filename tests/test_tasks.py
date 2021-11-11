import pytest

from workload import Job, Resource, TaskState, Task


def __create_default_task(job=Job(name="Perception"),
                          resource_requirement=Resource(name="CPU"),
                          runtime=1.0, deadline=10.0):
    """ Helper function to create a default task. """
    return Task(name="{}_Task".format(job.name), job=job,
                resource_requirement=resource_requirement,
                runtime=runtime, deadline=deadline)


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
