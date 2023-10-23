import flowlessly_py

def print_flow_graph(graph):
    min_cost = 0
    # Assuming you have methods or direct access to max_node_id_, arcs_ in the python class.
    # You can get those values as you exposed them using pybind11.
    for node_id in range(1, graph.get_max_node_id() + 1):
        for id_other_node, arc in graph.get_arcs()[node_id].items():
            if arc.is_fwd:
                flow = arc.reverse_arc.residual_cap + arc.min_flow
                if flow > 0:
                    print(f"flow from {node_id} to {id_other_node} is {flow}")
                    min_cost += flow * arc.cost

    print(f"cost: {min_cost}")

stats = flowlessly_py.Statistics()
graph = flowlessly_py.AdjacencyMapGraph(stats)

# graph.ReadGraph("/workspaces/flow-scheduler-simulator/erdos-scheduling-simulator/schedulers/Flowlessly/test_graphs/small_graph.in", True, False)

"""
Setup the following graph.
      4 \
         3 - 2 - 1
      5 /
"""

node_id = graph.AddNode(-1, 0, flowlessly_py.NodeType.OTHER, False)
assert node_id == 1
node_id = graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)
assert node_id == 2
node_id = graph.AddNode(0, 0, flowlessly_py.NodeType.OTHER, False)
assert node_id == 3
node_id = graph.AddNode(1, 0, flowlessly_py.NodeType.OTHER, False)
assert node_id == 4
node_id = graph.AddNode(1, 0, flowlessly_py.NodeType.OTHER, False)
assert node_id == 5

graph.AddArc(4, 3, 0, 1, 1, 0)
graph.AddArc(5, 3, 0, 1, 2, 0)
graph.AddArc(3, 2, 0, 2, 1, 0)
graph.AddArc(2, 1, 0, 2, 0, 0)

solver = flowlessly_py.SuccessiveShortest(graph, stats)
solver.Run()

# for i, elem in enumerate(graph.get_arcs()):
#     print(i, elem)

print_flow_graph(graph)