// cppimport
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class ERDOSScheduler {
    private:
        bool preemptive;
        uint64_t runtime;

    public:
        ERDOSScheduler(bool preemptive, uint64_t runtime): preemptive(preemptive), runtime(runtime) {
            std::cout << "initialized";
        }
};

PYBIND11_MODULE(erdos_scheduler_cpp, m) {
    py::class_<ERDOSScheduler>(m, "ERDOSScheduler")
        .def(py::init<bool, uint64_t>());
}

/*
<%
setup_pybind11(cfg)
%>
*/
