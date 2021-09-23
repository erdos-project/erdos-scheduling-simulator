from copy import deepcopy
from typing import List, Tuple
import time
import math

from workers.worker_pool import WorkerPool
from workload.lattice import Lattice
from workload.task import Task
from enum import Enum


class EventType(Enum):
    SCHEDULERSTART = 0
    SCHEDULERFINISHED = 1
    TASKRELEASE = 2
    TASKFINISHED = 3
    END = 4


class Event:
    def __init__(self,
                 type: EventType,
                 time,
                 task: Task = None,
                 sched_actions: List[Tuple[int, int]] = None):
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


class Workload(object):
    def __init__(
        self,
        needs_gpu: List[bool],
        release_times: List[int],
        absolute_deadlines: List[int],
        expected_runtimes: List[int],
        dependency_matrix,
        pinned_tasks: List[int],
    ):
        self.needs_gpu = needs_gpu
        self.release_times = release_times
        self.absolute_deadlines = absolute_deadlines
        self.expected_runtimes = expected_runtimes
        self.dependency_matrix = dependency_matrix
        self.pinned_tasks = pinned_tasks

    # IMO, the workload should be a wrapper around a list of Task. 
    # Each Task has a release time, an absolute deadline, runtime, 
    # and exposes needs_gpu and get_dependencies methods. We don't 
    # have to do this change in this PR, but we should do it in a future PR.

    # I'm suggesting this change to make the code a bit more modular, and to 
    # reduce the error-prone assumptions we're making (e.g., tasks have sequentially 
    # increasing ids, which we use to access lists, etc.,).


class Resources(object):
    def __init__(
            self,
            running_tasks: List[
                int],  # indicates the worker it's running on None otherwise
            num_tasks: int,
            num_gpus: int,
            num_cpus: int):
        self.running_tasks = running_tasks
        self.num_tasks = num_tasks
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus

    # IMO, this should be a Dict of TaskIds. Moreover, the Resources class should expose 
    # start_task(task: Task) and end_task(task_id: TaskId) methods, which can do any necessary 
    # booking (e.g., update running tasks, updated the available number of GPUs, yada yada).

    # Moreover, the Resource class might expose get_num_available_resource(resource: ResourceType).

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
        self.is_scheduling = False
        self.double_scheduled = False
    # num_cpu and num_gpu not used
    # IMO, the Simulator should receive a List[Task], which in our case 
    # are actually jobs that release tasks, Resources, and instant_scheduling.
    # It's unclear why the simulator needs to receive a Lattice.

    def schedule(self, workload: Workload, resources: Resources):
        raise NotImplementedError

    def simulate(self, timeout: int, v=0):
        def profile_scheduler():
            # schedule
            if self.is_scheduling:
                print("WARNING: Double Schedule")

            else:
                sched_actions, sched_time = self.run_scheduler()
                new_events = [
                    Event(EventType.SCHEDULERFINISHED,
                          self.time + sched_time,
                          sched_actions=sched_actions)
                ]
                event_queue.extend(new_events)
                self.is_scheduling = True

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

            if event.type == EventType.TASKFINISHED:

                # add new tasks
                new_task_queue = self.task_finish()
                new_event_queue = [
                    Event(EventType.TASKRELEASE, t.release_time, task=t)
                    for t in new_task_queue
                ]
                event_queue.extend(new_event_queue)

            if (event.type == EventType.TASKRELEASE
                    or event.type == EventType.TASKFINISHED):
                # next time step trigger scheduling
                if not self.is_scheduling:
                    event_queue.append(
                        Event(EventType.SCHEDULERSTART, self.time + 1))
                else:
                    self.double_scheduled = True
            if event.type == EventType.SCHEDULERSTART:
                profile_scheduler()

            if event.type == EventType.SCHEDULERFINISHED:
                # place tasks
                task_finished = self.place_jobs(event.sched_actions)
                task_finished = [
                    Event(EventType.TASKFINISHED,
                          t.time_remaining + self.time,
                          task=t) for t in task_finished
                ]
                event_queue.extend(task_finished)
                self.is_scheduling = False
                if self.double_scheduled:
                    event_queue.append(
                        Event(EventType.SCHEDULERSTART, self.time + 1))
                    # allows the updates in the time cycle to
                    # take effect before running scheduler again

    def advance_clock(self, step_size):

        self.time += step_size
        for worker in self.worker_pool.workers():
            if worker.current_task:
                worker.current_task.step(step_size=step_size)
    # IMO, this should go into the task_end method I mentioned in a comment above. 
    # We still need to do some reconciling between workers and resources. 
    # We might want to merge the Resources and Worker classes so that 
    # we only have one class that describes and manages a clump of related resources.

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
        workload = Workload(
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
        )
        resources = Resources(
            [worker for worker, _ in runnable
             ],  # indicates the worker it's running on None otherwise
            len(runnable),
            self.worker_pool.num_gpus,
            self.worker_pool.num_cpus)
        start_time = time.time()
        placement, assignment = self.schedule(workload, resources)
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

    def schedule(self, workload, resources):

        needs_gpu = workload.needs_gpu
        absolute_deadlines = workload.absolute_deadlines
        running_tasks = resources.running_tasks
        num_tasks = resources.num_tasks
        num_gpus = resources.num_gpus
        num_cpus = resources.num_cpus

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

    def schedule(self, workload, resources):
        num_gpus = resources.num_gpus
        num_cpus = resources.num_cpus
        running_tasks = resources.running_tasks
        needs_gpu = workload.needs_gpu

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
