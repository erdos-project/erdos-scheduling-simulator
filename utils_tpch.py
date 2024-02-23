# Code adopted from decima-sim to use their profiles of TPC-H queries

import numpy as np
import os

HOME_TPCH_DIR = "../profiles/workload/tpch_decima/"
TPCH_SUBDIR = "2g/"

class SetWithCount(object):
    """
    allow duplication in set
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]

def pre_process_task_duration(task_duration):
    # remove fresh durations from first wave
    clean_first_wave = {}
    for e in task_duration['first_wave']:
        clean_first_wave[e] = []
        fresh_durations = SetWithCount()
        # O(1) access
        for d in task_duration['fresh_durations'][e]:
            fresh_durations.add(d)
        for d in task_duration['first_wave'][e]:
            if d not in fresh_durations:
                clean_first_wave[e].append(d)
            else:
                # prevent duplicated fresh duration blocking first wave
                fresh_durations.remove(d)


def get_all_stage_info_for_query(query_num):
    dirname = os.path.dirname(__file__)
    task_durations = np.load(os.path.join(HOME_TPCH_DIR, TPCH_SUBDIR,'task_duration_' + str(query_num) + '.npy'), 
                            allow_pickle=True).item()

    num_nodes = len(task_durations)
    # nodes = []

    stage_info = {}

    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration['first_wave'])) 
        # NOTE: somehow only picks the first element {2: [n_tasks_in_ms]}

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l])

        # generate tasks in a node
        # tasks = []
        # for j in range(num_tasks):
        #     # task = Task(j, rough_duration, wall_time)
        #     tasks.append([j, rough_duration])

        # generate a node
        # print("created a node: ", n, " with tasks: ", tasks, " , task_duration: ", task_duration)
        # node = Node(n, tasks, task_duration, wall_time, np_random)
        # nodes.append([n, tasks, task_duration])

        curr_stage = {"stage_id": n, "num_tasks": num_tasks, "avg_task_duration": round(rough_duration)}
        stage_info[n] = curr_stage
    
    return stage_info