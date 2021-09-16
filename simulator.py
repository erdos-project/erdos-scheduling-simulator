from copy import deepcopy
from typing import List
import time
import math

from workers.worker_pool import WorkerPool
from workload.lattice import Lattice
from workload.task import Task
from enum import Enum


class EventType(Enum):
    SCHEDULERFINISHED = 1
    TASKRELEASE = 2
    TASKFINISHED = 3
    END = 4


class Event:
    def __init__(self, type: EventType, time, task=None, sched_actions=None):
        self.type = type
        self.time = time
        self.task = task
        self.sched_actions = sched_actions
        assert (self.type != EventType.TASKRELEASE or self.task is not None)
        assert (self.type != EventType.SCHEDULERFINISHED
                or self.sched_actions is not None)

    def __repr__(self):
        return str(self.type.name)

    def __lt__(self, other):
        return (self.time, self.type.value) < (other.time, other.type.value)


class Simulator:
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: Lattice,
                 instant_scheduling=False):
        """
        Args:
            tasks_list: List of Tasks.
            lattice: Default one would just have
                operators not connected to each other
        """
        self.tasks_list = tasks_list.copy()
        self.worker_pool = WorkerPool(num_cpus, num_gpus)
        self.lattice = deepcopy(lattice)
        self.current_time = 0
        self.current_tasks = []
        self.time = 0
        self.pending = []
        self.instant_scheduling = instant_scheduling
        self.am_scheduling = False

    def schedule(
            self,
            needs_gpu: List[bool],
            release_times: List[int],
            absolute_deadlines: List[int],
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int],
            running_tasks: List[
                int],  # indicates the worker it's running on None otherwise
            num_tasks: int,
            num_gpus: int,
            num_cpus: int):
        raise NotImplementedError

    def simulate(self, timeout: int, v=0):
        if not self.tasks_list:
            return
        self.tasks_list.sort(key=(lambda e: e.release_time))
        self.time = 0

        event_queue = [
            Event(EventType.TASKRELEASE, t.release_time, task=t)
            for t in self.tasks_list
        ]
        event_queue.append(Event(EventType.END, timeout))

        # implement as a heap
        while True:

            event_queue.sort()
            # import pdb; pdb.set_trace()
            # check what happens if there are
            # concurrent events (ordering of event.type)
            assert (len(event_queue) != 0), "No Event Queue"
            event = event_queue.pop(0)
            self.advance_clock(event.time - self.time)
            if event.type == EventType.END:
                break
            if event.type == EventType.TASKRELEASE:

                # add to pending
                self.pending.append(event.task)
                print("Activate: {}".format(event.task))

                # schedule
                if self.am_scheduling:
                    print("WARNING: Double Schedule")

                else:

                    sched_actions, sched_time = self.run_scheduler()
                    new_events = [
                        Event(EventType.SCHEDULERFINISHED,
                              self.time + sched_time,
                              sched_actions=sched_actions)
                    ]
                    event_queue.extend(new_events)
                    self.am_scheduling = True

            if event.type == EventType.TASKFINISHED:

                # add new tasks
                new_task_queue = self.task_finish()
                new_event_queue = [
                    Event(EventType.TASKRELEASE, t.release_time, task=t)
                    for t in new_task_queue
                ]
                event_queue.extend(new_event_queue)

                # schedule
                if self.am_scheduling:
                    print("WARNING: Double Schedule")

                else:
                    sched_actions, sched_time = self.run_scheduler()
                    new_events = [
                        Event(EventType.SCHEDULERFINISHED,
                              self.time + sched_time,
                              sched_actions=sched_actions)
                    ]
                    event_queue.extend(new_events)
                    self.am_scheduling = True

            if event.type == EventType.SCHEDULERFINISHED:
                # place tasks
                task_finished = self.place_jobs(event.sched_actions)
                task_finished = [
                    Event(EventType.TASKFINISHED,
                          t.time_remaining + self.time,
                          task=t) for t in task_finished
                ]
                event_queue.extend(task_finished)
                self.am_scheduling = False

    def advance_clock(self, step_size):

        self.time += step_size
        for worker in self.worker_pool.workers():
            if worker.current_task:
                worker.current_task.step(step_size=step_size)

    def task_finish(self):
        new_task_queue = []
        for worker in self.worker_pool.workers():
            if worker.current_task:
                if worker.current_task.time_remaining == 0:
                    worker.current_task.finish(new_task_queue, self.lattice,
                                               self.time)
                    print("Finished task: {}".format(worker.current_task))
                    worker.current_task = None
        return new_task_queue

    def run_scheduler(self):
        running_tasks = self.worker_pool.get_running_tasks()
        runnable = running_tasks + [(None, t) for t in self.pending]
        start_time = time.time()
        placement, assignment = self.schedule(
            [t.needs_gpu for _, t in runnable],
            [
                t.release_time -
                self.time if t.release_time is not None else None
                for _, t in runnable
            ],
            [
                t.deadline - self.time if t.deadline is not None else None
                for _, t in runnable
            ],
            [t.time_remaining for _, t in runnable],
            None,  # dependencies
            [None for _, t in runnable],  # pinned
            [worker for worker, _ in runnable
             ],  # indicates the worker it's running on None otherwise
            len(runnable),
            self.worker_pool.num_gpus,
            self.worker_pool.num_cpus)
        end_time = time.time()
        task_queue = []
        schedule_actions = []  # for now just indicates what should be added
        for i, (p, a) in enumerate(zip(placement, assignment)):
            _, t = runnable[i]
            if a == 0:
                w = self.worker_pool.get_worker(p)
                if (w.current_task is None
                        or w.current_task.unique_id != t.unique_id):
                    schedule_actions.append((w.unique_id, t))
            else:
                task_queue.append(t)
        # task_queue use to be the new pending list
        scheduling_overhead = math.ceil(1000 * (end_time - start_time))
        if self.instant_scheduling:
            scheduling_overhead = 1
        return schedule_actions, scheduling_overhead

    def place_jobs(self, schedule_actions):

        for w_id, t in schedule_actions:
            w = self.worker_pool.get_worker(w_id)
            if (w.current_task is not None):
                import pdb
                pdb.set_trace()
            assert (w.current_task is None), "Attempting to preempt"
            w.do_job(t, self.lattice, self.time)
        added_tasks = [t for _, t in schedule_actions]
        added_tasks_id = [t.unique_id for t in added_tasks]
        self.pending = list(
            filter(lambda t: t.unique_id not in added_tasks_id, self.pending))
        return added_tasks

    def old_sim(self, tasks_list, timeout, v=0):
        task_queue = tasks_list
        while self.time < timeout:
            if v > 0 and self.time % 100 == 0:
                print("step: {}".format(self.time))
            if v > 1:
                print("step: {}".format(self.time))
            # first determine if there's new tasks to be made visible
            while len(self.tasks_list
                      ) != 0 and self.tasks_list[0].release_time == self.time:
                task = self.tasks_list.pop(0)
                print("Activate: {}".format(task))
                task_queue.append(task)
            # second determine where to place tasks
            # self.schedule(time, task_queue, timeout)
            running_tasks = self.worker_pool.get_running_tasks()
            runnable = running_tasks + [(None, t) for t in task_queue]
            # print (runnable)
            placement, assignment = self.schedule(
                [t.needs_gpu for _, t in runnable],
                [
                    t.release_time -
                    self.time if t.release_time is not None else None
                    for _, t in runnable
                ],
                [
                    t.deadline - self.time if t.deadline is not None else None
                    for _, t in runnable
                ],
                [t.time_remaining for _, t in runnable],
                None,  # dependencies
                [None for _, t in runnable],  # pinned
                [worker for worker, _ in runnable
                 ],  # indicates the worker it's running on None otherwise
                len(runnable),
                self.worker_pool.num_gpus,
                self.worker_pool.num_cpus)

            # place jobs task on worker
            task_queue = []
            for i, (p, a) in enumerate(zip(placement, assignment)):
                _, t = runnable[i]
                if a == 0:
                    w = self.worker_pool.get_worker(p)
                    if (w.current_task is None
                            or w.current_task.unique_id != t.unique_id):
                        w.do_job(t, self.lattice, self.time)
                else:
                    task_queue.append(t)

            # determine step_size as next release event or time remaining
            step_size, step_size_rel = None, None
            if len(self.worker_pool.get_running_tasks()) > 0:
                step_size = min([
                    t.time_remaining
                    for _, t in self.worker_pool.get_running_tasks()
                ])

            if self.worker_pool.has_free_worker() and len(self.tasks_list) > 0:
                step_size_rel = self.tasks_list[0].release_time - self.time
                if step_size is None:
                    step_size = step_size_rel
                else:
                    step_size = min(step_size, step_size_rel)
            if step_size_rel is None and step_size is None:
                if len(self.worker_pool.get_running_tasks()) == 0 and len(
                        self.tasks_list) == 0:
                    step_size = timeout - self.time
                else:
                    step_size = 1

            # finally advance the workers and time
            new_task_queue = []
            for worker in self.worker_pool.workers():
                if worker.current_task:
                    worker.current_task.step(step_size=step_size)
                    if worker.current_task.time_remaining == 0:
                        worker.current_task.finish(new_task_queue,
                                                   self.lattice,
                                                   self.time + step_size)
                        print("Finished task: {}".format(worker.current_task))
                        worker.current_task = None
            task_queue.extend(new_task_queue)  # add newly spawned tasks
            self.time += step_size

    def show_running_tasks(self):
        for worker in self.worker_pool.workers():
            if worker.current_task:
                print(f"Worker {worker.unique_id}: {worker.current_task}")

    def history(self):
        return self.worker_pool.history()


class EdfSimulator(Simulator):
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: Lattice,
                 preemptive: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.preemptive = preemptive

    def schedule(
            self,
            needs_gpu: List[bool],
            release_times: List[int],
            absolute_deadlines: List[int],
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int],
            running_tasks: List[
                int],  # indicates the worker it's running on None otherwise
            num_tasks: int,
            num_gpus: int,
            num_cpus: int):

        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,
                              num_cpus + num_gpus))  # cpus indexed after gpus
        placements = [None] * num_tasks
        assignment = [None] * num_tasks
        priority = [
            absolute_deadlines.index(e) for e in sorted(absolute_deadlines)
        ]

        if not self.preemptive:
            # check and place running tasks first
            placements = running_tasks
            assignment = [
                0 if (t is not None) else None for t in running_tasks
            ]
            gpu_pool = [i for i in gpu_pool if i not in running_tasks]
            cpu_pool = [i for i in cpu_pool if i not in running_tasks]
            # print (assignment, running_tasks)
            # print (gpu_count, cpu_count)
        for index in priority:
            if needs_gpu[index]:
                if len(gpu_pool) > 0 and running_tasks[index] is None:
                    placements[index] = gpu_pool.pop(0)
                    assignment[index] = 0
            else:
                if len(cpu_pool) > 0 and running_tasks[index] is None:
                    placements[index] = cpu_pool.pop(0)
                    assignment[index] = 0
            if len(gpu_pool) + len(cpu_pool) == 0:
                break

        return placements, assignment


class FifoSimulator(Simulator):
    def __init__(self,
                 num_cpus: int,
                 num_gpus: int,
                 tasks_list: List[Task],
                 lattice: Lattice,
                 gpu_exact_match: bool = False):
        super().__init__(num_cpus, num_gpus, tasks_list, lattice)
        self.gpu_exact_match = gpu_exact_match

    def schedule(
            self,
            needs_gpu: List[bool],
            release_times: List[int],
            absolute_deadlines: List[int],
            expected_runtimes: List[int],
            dependency_matrix,
            pinned_tasks: List[int],
            running_tasks: List[
                int],  # indicates the worker it's running on None otherwise
            num_tasks: int,
            num_gpus: int,
            num_cpus: int):

        gpu_pool = list(range(num_gpus))
        cpu_pool = list(range(num_gpus,
                              num_cpus + num_gpus))  # cpus indexed after gpus
        placements = running_tasks
        assignment = [0 if (t is not None) else None for t in running_tasks]
        gpu_pool = [i for i in gpu_pool if i not in running_tasks]
        cpu_pool = [i for i in cpu_pool if i not in running_tasks]

        for i, gpu in enumerate(needs_gpu):
            if gpu:
                if len(gpu_pool) > 0 and running_tasks[i] is None:
                    w_idx = gpu_pool.pop(0)
                    placements[i] = w_idx
                    assignment[i] = 0
            else:
                if len(cpu_pool) > 0 and running_tasks[i] is None:
                    w_idx = cpu_pool.pop(0)
                    placements[i] = w_idx
                    assignment[i] = 0
            if len(gpu_pool) + len(cpu_pool) == 0:
                break

        return placements, assignment
