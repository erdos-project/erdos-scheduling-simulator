## DAGSched

The package provides an implementation of the DAGSched scheduler. DAGSched is a declarative, mathematical optimization backed scheduler that is both deadline and DAG-aware. The interface to DAGSched is defined by the Space-Time Request Language (STRL) proposed by Tumanov et. al. in [EuroSys '16](https://dl.acm.org/doi/pdf/10.1145/2901318.2901355).

## Installation

DAGSched requires a backend solver to be able to solve the optimization problems generated for the scheduling instance. Currently, DAGSched supports [Gurobi](https://www.gurobi.com), [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) and [Google OR-Tools](https://developers.google.com/optimization). However, the Gurobi backend is preferred as it is most extensively tested in our daily use. Academic users can request a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/), and please note that the free license that ships with `gurobipy` is not sufficient to solve a moderately sized scheduling problem.

The installation requires a C++-20 compatible compiler and is done through CMake. Users can choose to set the following variables during installation:

| Flag            | Description |
| -----------     | ----------- |
| TETRISCHED_PERF_ENABLED | If ON, the backend will generate CSV files that can be used by `scripts/analyze_performance.py` to generate Chrome traces of the runtime of each part of the backend. |
| TETRISCHED_LOGGING_ENABLED | If ON, the backend will generate verbose logging files that can be used for debugging the execution of the backend on a given scheduling problem. |
| GUROBI_DIR | (Environment Variable) If provided, the Gurobi installation is linked to from this directory. Otherwise, the backend searches for a Gurobi installation, and if not found, does not compile the Gurobi specific features of the scheduler.
| TETRISCHED_GUROBI_VER | The version of the Gurobi installation to be compiled and linked with. Defaults to Gurobi 10.0. |
| CPLEX_DIR | (Environment Variable) If provided, the CPLEX installation is linked to from this directory. Otherwise, the backend searches for a CPLEX installation, and if not found, does not compile the CPLEX specific features of the scheduler. |
| ORTOOLS_DIR | (Environment Variable) If provided, the OR-Tools installation is linked to from this directory. Otherwise, the backend searches for a OR-Tools installation, and if not found, does not compile the OR-Tools specific features of the scheduler. |

To install the scheduler with a Gurobi backend, use the following instructions:

```bash
export GUROBI_DIR=/path/to/gurobi/dir/
export CMAKE_INSTALL_MODE=ABS_SYMLINK
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/path/to/compiler -DINSTALL_GTEST=OFF -DTBB_INSTALL=OFF -DTETRISCHED_GUROBI_VER=10 ..
make -j install
```

To test that the scheduler has successfully compiled and linked, run the test suite using:
```bash
./test_tetrisched
```


## Interfaces

DAGSched provides both C++ and Python interfaces. The `python` directory provides typing stubs that the users can link into their IDE of choice to get type hinting and documentation for the scheduler. For a reference implementation of the scheduler, check `schedulers/tetrisched_scheduler.py` that uses the Python interface to invoke the C++-based scheduling backend.
