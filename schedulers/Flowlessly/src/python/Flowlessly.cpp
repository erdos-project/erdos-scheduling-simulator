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

    py::class_<flowlessly::AdjacencyMapGraph>(flowlessly_m, "AdjacencyMapGraph")
        .def(py::init<flowlessly::Statistics*>())
        .def("ReadGraph", static_cast<void (flowlessly::AdjacencyMapGraph::*)(const std::string&, bool, bool&)> (&flowlessly::AdjacencyMapGraph::ReadGraph))
        .def("AddArc", &flowlessly::AdjacencyMapGraph::AddArc)
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
        .def("get_nodes", &flowlessly::AdjacencyMapGraph::get_nodes);

    py::class_<flowlessly::SuccessiveShortest>(flowlessly_m, "SuccessiveShortest")
        .def(py::init<flowlessly::AdjacencyMapGraph*, flowlessly::Statistics*>())
        .def("PrepareState", &flowlessly::SuccessiveShortest::PrepareState)
        .def("Run", &flowlessly::SuccessiveShortest::Run);
}