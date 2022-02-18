from typing import List


def verify_schedule(start_times, placements, resource_requirements,
                    release_times, absolute_deadlines, expected_runtimes,
                    dependency_matrix, num_resources):
    # Check if each task's start time is greater than its release time.
    assert all([s >= r for s, r in zip(start_times, release_times)
                ]), "not_valid_release_times"

    num_resources_ub = [
        sum(num_resources[0:i + 1]) for i in range(len(num_resources))
    ]
    num_resources_lb = [0] + [
        sum(num_resources[0:i + 1]) for i in range(len(num_resources) - 1)
    ]
    assert all([
        all([
            p < num_resources_ub[i] and num_resources_lb[i] <= p
            if resource_req else True
            for i, resource_req in enumerate(resource_req_arr)
        ]) for resource_req_arr, p in zip(resource_requirements, placements)
    ]), "not_valid_placement"

    # Check if all tasks finished before the deadline.
    assert all([
        (d >= s + e)
        for d, e, s in zip(absolute_deadlines, expected_runtimes, start_times)
    ]), "doesn't finish before deadline"

    # Check if the task dependencies were satisfied.
    for i, row_i in enumerate(dependency_matrix):
        for j, column_j in enumerate(row_i):
            if i != j and column_j:
                assert start_times[i] + expected_runtimes[i] <= start_times[
                    j], f"not_valid_dependency{i}->{j}"

    # Check if tasks overlapped on a resource.
    placed_tasks = [
        (p, s, s + e)
        for p, s, e in zip(placements, start_times, expected_runtimes)
    ]
    placed_tasks.sort()
    for t1, t2 in zip(placed_tasks, placed_tasks[1:]):
        if t1[0] == t2[0]:
            assert t1[2] <= t2[1], f"overlapping_tasks_on_{t1[0]}"


def compute_slack_cost(placement, expected_runtime, absolute_deadlines):
    slacks = [
        d - e - p
        for p, e, d in zip(placement, expected_runtime, absolute_deadlines)
    ]
    return sum(slacks)


def needs_gpu_to_resource_requirements(needs_gpu: List[bool]):
    return [[False, True] if need_gpu else [True, False]
            for need_gpu in needs_gpu]
