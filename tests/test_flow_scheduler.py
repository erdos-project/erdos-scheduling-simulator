import flowlessly_py
from schedulers.flow_scheduler import extract_placement_from_flow_graph, populate_min_cost_max_flow_graph
from tests.utils import create_default_task
from workers import Worker, WorkerPool, WorkerPools
from workload import (
    Job,
    Resource,
    Resources,
)

def test_populate_min_cost_max_flow_graph():
    task_a = create_default_task(
        job=Job(name="Job_A"), deadline=10, runtime=500,
    )
    task_b = create_default_task(
        job=Job(name="Job_B"), deadline=5, runtime=1000,
    )
    task_c = create_default_task(
        job=Job(name="Job_C"), deadline=1, runtime=1500,
    )

    tasks_to_be_scheduled = [task_a, task_b, task_c]
    task_id_to_execution_run_time = {
        task_a.id: 500,
        task_b.id: 1000,
        task_c.id: 1500,
    }
    
    worker_a = Worker("Worker_A", Resources(resource_vector={Resource(name="CPU", _id="any"): 1}))
    worker_b = Worker("Worker_B", Resources(resource_vector={Resource(name="CPU", _id="any"): 1}))
    worker_pools = WorkerPools([WorkerPool("WorkerPool", [worker_a, worker_b])]) 

    flow_stats = flowlessly_py.Statistics()
    flow_graph = flowlessly_py.AdjacencyMapGraph(flow_stats)

    # Call function
    node_id_to_task_id, node_id_to_worker_id = populate_min_cost_max_flow_graph(flow_graph, tasks_to_be_scheduled, worker_pools)
    
    # Make sure each task is connect to each worker with right cost and capacity
    for node_id_task, task_id in node_id_to_task_id.items():
        for node_id_worker in node_id_to_worker_id:
            assert node_id_worker in flow_graph.get_arcs()[node_id_task].keys()
            flow_graph.get_arcs()[node_id_task][node_id_worker].cost == task_id_to_execution_run_time[task_id]
            flow_graph.get_arcs()[node_id_task][node_id_worker].residual_cap == 1
            
def test_extract_placement_from_flow_graph():
    task_a = create_default_task(
        job=Job(name="Job_A"), deadline=10, runtime=500,
    )
    task_b = create_default_task(
        job=Job(name="Job_B"), deadline=5, runtime=1000,
    )
    task_c = create_default_task(
        job=Job(name="Job_C"), deadline=1, runtime=1500,
    )
    tasks_to_be_scheduled = [task_a, task_b, task_c]
    
    worker_a = Worker("Worker_A", Resources(resource_vector={Resource(name="RAM", _id="any"): 100, Resource(name="CPU", _id="any"): 1}))
    worker_b = Worker("Worker_B", Resources(resource_vector={Resource(name="RAM", _id="any"): 100, Resource(name="CPU", _id="any"): 1}))
    worker_pools = WorkerPools([WorkerPool("WorkerPool", [worker_a, worker_b])]) 

    flow_stats = flowlessly_py.Statistics()
    flow_graph = flowlessly_py.AdjacencyMapGraph(flow_stats)

    node_id_to_task_id, node_id_to_worker_id = populate_min_cost_max_flow_graph(flow_graph, tasks_to_be_scheduled, worker_pools)
    
    solver = flowlessly_py.SuccessiveShortest(flow_graph, flow_stats)
    solver.Run()

    task_id_to_worker_id = extract_placement_from_flow_graph(flow_graph, node_id_to_task_id, node_id_to_worker_id)

    assert task_a.id in task_id_to_worker_id
    assert task_b.id in task_id_to_worker_id
    assert task_c.id not in task_id_to_worker_id

    assert task_id_to_worker_id[task_a.id] != task_id_to_worker_id[task_b.id]
    assert task_id_to_worker_id[task_a.id] in [worker_a.id, worker_b.id]
    assert task_id_to_worker_id[task_b.id] in [worker_a.id, worker_b.id]