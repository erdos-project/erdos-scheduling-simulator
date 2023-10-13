#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Required for converting STL types to Python types

#include <boost/thread/latch.hpp>

#include "graphs/graph.h"
#include "graphs/adjacency_map_graph.h"
#include "solvers/successive_shortest.h"

DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");

namespace py = pybind11;

PYBIND11_MODULE(flowlessly_py, flowlessly_m) {
    flowlessly_m.doc() = "Python API for Flowlessly."; // optional module docstring

    py::class_<flowlessly::Statistics>(flowlessly_m, "Statistics")
        .def(py::init<>());

    // Define the NodeType enum.
    py::enum_<flowlessly::NodeType>(flowlessly_m, "NodeType")
        .value("OTHER", flowlessly::NodeType::OTHER)
        .value("TASK", flowlessly::NodeType::TASK)
        .value("PU", flowlessly::NodeType::PU)
        .value("SINK", flowlessly::NodeType::SINK)
        .value("MACHINE", flowlessly::NodeType::MACHINE)
        .value("INTERMEDIATE_RES", flowlessly::NodeType::INTERMEDIATE_RES)
        .export_values();
    
    py::enum_<flowlessly::NodeStatus>(flowlessly_m, "NodeStatus")
        .value("NOT_VISITED", flowlessly::NodeStatus::NOT_VISITED)
        .value("VISITING", flowlessly::NodeStatus::VISITING)
        .value("VISITED", flowlessly::NodeStatus::VISITED)
        .export_values();

    py::class_<flowlessly::Node>(flowlessly_m, "Node")
        .def(py::init<>())  // Default constructor
        .def(py::init<const flowlessly::Node &>())  // Copy constructor
        .def_readwrite("supply", &flowlessly::Node::supply)
        .def_readwrite("potential", &flowlessly::Node::potential)
        .def_readwrite("distance", &flowlessly::Node::distance)
        .def_readwrite("type", &flowlessly::Node::type)
        .def_readwrite("status", &flowlessly::Node::status);
    
    py::class_<flowlessly::Arc>(flowlessly_m, "Arc")
        .def(py::init<>())  // Default constructor
        .def(py::init<uint32_t, uint32_t, bool, bool, int32_t, int32_t, int64_t, flowlessly::Arc*>())  // Parameterized constructor
        .def(py::init<const flowlessly::Arc&>())  // Copy constructor
        .def(py::init<flowlessly::Arc*>())  // Constructor from a pointer
        .def("CopyArc", &flowlessly::Arc::CopyArc)
        .def_readwrite("src_node_id", &flowlessly::Arc::src_node_id)
        .def_readwrite("dst_node_id", &flowlessly::Arc::dst_node_id)
        .def_readwrite("is_alive", &flowlessly::Arc::is_alive)
        .def_readwrite("is_fwd", &flowlessly::Arc::is_fwd)
        .def_readwrite("is_running", &flowlessly::Arc::is_running)
        .def_readwrite("residual_cap", &flowlessly::Arc::residual_cap)
        .def_readwrite("min_flow", &flowlessly::Arc::min_flow)
        .def_readwrite("cost", &flowlessly::Arc::cost)
        .def_readwrite("reverse_arc", &flowlessly::Arc::reverse_arc);

    py::class_<flowlessly::AdjacencyMapGraph>(flowlessly_m, "AdjacencyMapGraph")
        .def(py::init<flowlessly::Statistics*>())
        .def("ReadGraph", static_cast<void (flowlessly::AdjacencyMapGraph::*)(const std::string&, bool, bool&)> (&flowlessly::AdjacencyMapGraph::ReadGraph))
        .def("AddArc", &flowlessly::AdjacencyMapGraph::AddArc, pybind11::return_value_policy::reference_internal)
        .def("AddNode", static_cast<void (flowlessly::AdjacencyMapGraph::*)(uint32_t, int32_t, int64_t, flowlessly::NodeType, bool)> (&flowlessly::AdjacencyMapGraph::AddNode)) // Overloaded function
        .def("AddNode", static_cast<uint32_t (flowlessly::AdjacencyMapGraph::*)(int32_t, int64_t, flowlessly::NodeType, bool)> (&flowlessly::AdjacencyMapGraph::AddNode)) // Overloaded function
        .def("ChangeArc", static_cast<void (flowlessly::AdjacencyMapGraph::*)(flowlessly::Arc*, uint32_t, int32_t, int64_t, int32_t)> (&flowlessly::AdjacencyMapGraph::ChangeArc)) // Overloaded function
        .def("ChangeArc", static_cast<void (flowlessly::AdjacencyMapGraph::*)(uint32_t, uint32_t, uint32_t, int32_t, int64_t, int32_t, bool, int64_t)> (&flowlessly::AdjacencyMapGraph::ChangeArc)) // Overloaded function
        .def("GetMachinePUs", &flowlessly::AdjacencyMapGraph::GetMachinePUs)
        .def("GetRandomArc", &flowlessly::AdjacencyMapGraph::GetRandomArc)
        .def("GetTaskAssignments", &flowlessly::AdjacencyMapGraph::GetTaskAssignments)
        .def("GetTotalCost", &flowlessly::AdjacencyMapGraph::GetTotalCost)
        .def("InitializeGraph", &flowlessly::AdjacencyMapGraph::InitializeGraph)
        .def("IsEpsOptimal", &flowlessly::AdjacencyMapGraph::IsEpsOptimal)
        .def("IsFeasible", &flowlessly::AdjacencyMapGraph::IsFeasible)
        .def("IsInTopologicalOrder", &flowlessly::AdjacencyMapGraph::IsInTopologicalOrder)
        .def("RemoveArc", &flowlessly::AdjacencyMapGraph::RemoveArc)
        .def("RemoveNode", &flowlessly::AdjacencyMapGraph::RemoveNode)
        .def("WriteAssignments", &flowlessly::AdjacencyMapGraph::WriteAssignments)
        .def("WriteFlowGraph", &flowlessly::AdjacencyMapGraph::WriteFlowGraph)
        .def("WriteGraph", &flowlessly::AdjacencyMapGraph::WriteGraph)
        .def("get_arcs", &flowlessly::AdjacencyMapGraph::get_arcs, pybind11::return_value_policy::reference_internal)
        .def("get_nodes", &flowlessly::AdjacencyMapGraph::get_nodes, pybind11::return_value_policy::reference_internal)
        .def("get_max_node_id", &flowlessly::AdjacencyMapGraph::get_max_node_id)
        .def("get_sink_node", &flowlessly::AdjacencyMapGraph::get_sink_node)
        .def("get_source_nodes", &flowlessly::AdjacencyMapGraph::get_source_nodes, py::return_value_policy::reference_internal)
        .def("get_potentials", &flowlessly::AdjacencyMapGraph::get_potentials, py::return_value_policy::reference_internal);

    py::class_<flowlessly::SuccessiveShortest>(flowlessly_m, "SuccessiveShortest")
        .def(py::init<flowlessly::AdjacencyMapGraph*, flowlessly::Statistics*>())
        .def("LogStatistics", &flowlessly::SuccessiveShortest::LogStatistics)
        .def("ResetStatistics", &flowlessly::SuccessiveShortest::ResetStatistics)
        .def("Run", &flowlessly::SuccessiveShortest::Run);
}