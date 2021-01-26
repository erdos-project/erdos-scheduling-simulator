class Operator:
    def __init__(self,unique_id,children,time_required,is_join=False, relative_deadline=False):
#         print ("initialized operator {}".format(unique_id))
        self.children = children
        self.log = []
        self.unique_id = unique_id
        self.time_required = time_required
        self.is_join = is_join
        self.paused_job = None
        self.relative_deadline = relative_deadline
        
class Lattice:
    def __init__(self, operators):
        self.operators = operators
        self.children_dir = {}
        for op in operators:
            self.children_dir[op.unique_id] = op.children 
        # should prolly sort the operators 
    def __repr__(self):
        return "Lattice {}".format(self.children_dir.__repr__())            
    def get_children(self,operator):
        return self.children_dir[operator]
    def add_next(self,event,new_event_queue,curr_time):
        '''
            Adds new events to queue if necessary when an Event completes
            The times are a bit weird cause this effectively gets called at the end of the curr_time slice/beginning of the next slice
        '''
        children = self.get_children(event.operator)
        for c in children:
            c = self.get_op(c)
            event_id = event.unique_id
            if c.is_join:
                if not c.paused_job:
                    c.paused_job = event
                    continue
                else:
                    event_id = min(event.unique_id,c.paused_job.unique_id)
                    c.paused_job = None
            if c.relative_deadline:
                d = c.relative_deadline + curr_time + 1
            else: 
                d = None
            new_event_queue.append(Event(event_id,c.unique_id, curr_time + 1,deadline=d))
    def get_op(self,op):
        '''
            Assumes that the operators are in sorted and consecutive order
        '''
        out = self.operators[op]
        if out.unique_id != op:
            print ("ERROR: lattice not properly ordered [{} returned instead of {}]".format(out.unique_id,op))
        return out
class Event:
    def __init__(self,  unique_id, operator , arrival_time, deadline=None):
        self.unique_id = unique_id
        self.operator = operator
        self.arrival_time = arrival_time
        self.time_remaining = -1
        self.start_time = None
        self.finish_time = None
        self.deadline = deadline
    def __repr__(self):
        return "<Event {}; Running Op {}; Available at Time {}; Executed {} to {}; Deadline: {}>".format(self.unique_id,self.operator, self.arrival_time, self.start_time, self.finish_time, self.deadline)
    def start(self,lattice,time):
        self.time_remaining = lattice.get_op(self.operator).time_required
        self.start_time = time 
    def step(self):
        if self.time_remaining < 1:
            print("ERROR: already finished event; {} left".format(self.time_remaining))
        self.time_remaining -= 1
        return self.time_remaining
    def finish(self,new_event_queue,lattice,time):
        if self.time_remaining != 0:
            print("ERROR: try to finish but not done; {} left".format(self.time_remaining))
        lattice.add_next(self,new_event_queue,time)
        self.finish_time = time + 1 # cause you technically am using up the current time slice
        if self.deadline and self.deadline < self.finish_time:
            print("WARNING: DEADLINE MISSED [D:{}; F:{}]".format(self.deadline,self.finish_time))
        
        


class Worker: 
    def __init__(self, unique_id, gpus=1):
        self.unique_id = unique_id
        self.history = []
        self.current_event = None
    def __repr__(self):
        return "Worker {}: {}".format(self.unique_id, self.history)
    def do_job(self, task, lattice,time):
        if self.current_event != None:
            print("ERROR: Worker is still busy")
        self.history.append(task)
        self.current_event = task
        task.start(lattice,time)
    def reset(self):
        self.__init__(self.unique_id)
    def get_history(self):
        out = ""
        for e in self.history:
            out += "\n  {}".format(e)
        return "[{}\n]".format(out)
            
        
class WorkerPool:
    def __init__(self, workers):
        self.workers = workers
        self.count = len(workers)
    def __repr__(self):
        output = ""
        for w in self.workers:
            output += "\n{}".format(w)
        return "Worker Pool: " + output
    def add_worker(self,w):
        for v in self.workers:
            if v.unique_id == w.unique_id:
                print ("ERROR: worker not unique [{}]".format(w.unique_id))
                return 
        self.workers.append(w)
    def add_workers(self,ws):
        for w in ws:
            self.add_worker(w)
    def reset(self):
        for w in self.workers:
            w.reset()
    def history(self):
        output = ""
        for w in self.workers:
            output += "\nWorker{}: {}".format(w.unique_id, w.get_history())
        return "Worker Pool History: " + output
        
def simulate(schedule,task_set,worker_pool,lattice,timeout,v=0):
    if not task_set:
        return
    task_set.sort(key=(lambda e: e.arrival_time))
    event_queue = []
    for time in range(timeout):
        if v:
            print ("step: {}".format(time))
        while len(task_set) != 0 and  task_set[0].arrival_time == time:
            task = task_set.pop(0)
            print ("Activate: {}".format(task))
            event_queue.append(task)
        schedule(time,event_queue,lattice,worker_pool,timeout)
def fifo_schedule(time,event_queue,lattice,worker_pool,timeout):
    new_event_queue = []
#     print (event_queue)
    for worker in worker_pool.workers:
        if worker.current_event == None and event_queue:
            worker.do_job(event_queue.pop(0),lattice,time)
        if worker.current_event: 
            worker.current_event.step()
            if worker.current_event.time_remaining == 0:
                worker.current_event.finish(new_event_queue,lattice,time)
                print("finished event: {}".format(worker.current_event))
                worker.current_event = None
    event_queue.extend(new_event_queue)

def edf_schedule(time,event_queue,lattice,worker_pool,timeout):
    new_event_queue = []
#     print (event_queue)
    for worker in worker_pool.workers:
        if worker.current_event == None and event_queue:
            worker.do_job(event_queue.pop(0),lattice,time)
        if worker.current_event: 
            worker.current_event.step()
            if worker.current_event.time_remaining == 0:
                worker.current_event.finish(new_event_queue,lattice,time)
                print("finished event: {}".format(worker.current_event))
                worker.current_event = None
    event_queue.extend(new_event_queue)
    event_queue.sort(key=lambda x: x.deadline if x.deadline else timeout)
def llf_schedule(time,event_queue,lattice,worker_pool,timeout):
    new_event_queue = []
#     print (event_queue)
    for worker in worker_pool.workers:
        if worker.current_event == None and event_queue:
            worker.do_job(event_queue.pop(0),lattice,time)
        if worker.current_event: 
            worker.current_event.step()
            if worker.current_event.time_remaining == 0:
                worker.current_event.finish(new_event_queue,lattice,time)
                print("finished event: {}".format(worker.current_event))
                worker.current_event = None
    event_queue.extend(new_event_queue)
    event_queue.sort(key=lambda x: x.deadline - x.time_remaining if x.deadline else timeout)

