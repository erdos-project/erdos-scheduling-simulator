cmake_minimum_required(VERSION 3.5...3.26)
project(tetrisched LANGUAGES CXX)

# Compile with atleast C++20.
# Run export CMAKE_INSTALL_MODE=ABS_SYMBLINK to install dependencies correctly.
# Set cmake -D CMAKE_CXX_COMPILER=/path/to/C++20/compatible/compiler.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

# Include debug information.
set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/eab0e7e289db13eabfc246809b0284dac02a369d.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set the directories where the solver backends are installed.
if (DEFINED ENV{CPLEX_DIR})
    set(CPLEX_DIR ENV{CPLEX_DIR})
    message("-- Using CPLEX installation at: ${CPLEX_DIR}")
else()
    set(CPLEX_DIR "/opt/ibm/ILOG/CPLEX_Studio2211")
    message("-- Using default CPLEX installation at: ${CPLEX_DIR}")
endif()

if (DEFINED ENV{GUROBI_DIR})
   set(GUROBI_DIR ENV{GUROBI_DIR})
   message("-- Using Gurobi installation at: ${GUROBI_DIR}")
else()
   set(GUROBI_DIR "/opt/gurobi1001/linux64")
   message("-- Using default Gurobi installation at: ${GUROBI_DIR}")
endif()

if (DEFINED ENV{ORTOOLS_DIR})
   set(ORTOOLS_DIR ENV{ORTOOLS_DIR})
   message("-- Using Google OR-Tools installation at: ${ORTOOLS_DIR}")
else()
   set(ORTOOLS_DIR "/usr/local/include/ortools")
   message("-- Using default Google OR-Tools installation at: ${ORTOOLS_DIR}")
endif()

set(SOLVER_BACKEND_INCLUDE_DIRS "")
set(SOLVER_BACKEND_LINK_DIRS "")
set(SOLVER_BACKEND_LINK_LIBRARIES "")

if (EXISTS "${CPLEX_DIR}")
     message("-- Adding CPLEX : ${CPLEX_DIR}")
     add_compile_definitions(_TETRISCHED_WITH_CPLEX_)

     LIST(APPEND SOLVER_BACKEND_INCLUDE_DIRS
          "${CPLEX_DIR}/cplex/include/"
          "${CPLEX_DIR}/concert/include/")
     LIST(APPEND SOLVER_BACKEND_LINK_DIRS
          "${CPLEX_DIR}/cplex/lib/x86-64_linux/static_pic/"
          "${CPLEX_DIR}/concert/lib/x86-64_linux/static_pic/")
     LIST(APPEND SOLVER_BACKEND_LINK_LIBRARIES
          "ilocplex"
          "concert"
          "cplex"
          "m"
          "pthread"
          "dl")
else()
     message("-- Not Adding CPLEX")
endif()

if (EXISTS "${GUROBI_DIR}")
     message("-- Adding GUROBI : ${GUROBI_DIR}")
     add_compile_definitions(_TETRISCHED_WITH_GUROBI_)

     LIST(APPEND SOLVER_BACKEND_INCLUDE_DIRS
          "${GUROBI_DIR}/include/")
     LIST(APPEND SOLVER_BACKEND_LINK_DIRS
          "${GUROBI_DIR}/lib/")
     LIST(APPEND SOLVER_BACKEND_LINK_LIBRARIES
          "gurobi_c++"
          "gurobi100")
else()
     message("-- Not Adding GUROBI")
endif()

if (EXISTS "${ORTOOLS_DIR}")
     message("-- Adding Google OR-Tools : ${ORTOOLS_DIR}")
     add_compile_definitions(_TETRISCHED_WITH_OR_TOOLS_)

     LIST(APPEND SOLVER_BACKEND_LINK_LIBRARIES
	     "ortools")
else()
     message("-- Not adding OR-Tools")
endif()

set(TETRISCHED_SOURCE
     "src/Expression.cpp"
     "src/Partition.cpp"
     "src/SolverModel.cpp"
     "src/Scheduler.cpp")

if (EXISTS "${CPLEX_DIR}")
     LIST(APPEND TETRISCHED_SOURCE "src/CPLEXSolver.cpp")
endif()
if (EXISTS "${GUROBI_DIR}")
     LIST(APPEND TETRISCHED_SOURCE "src/GurobiSolver.cpp")
endif()
if (EXISTS "${ORTOOLS_DIR}")
     LIST(APPEND TETRISCHED_SOURCE "src/GoogleCPSolver.cpp")
endif()

add_library(tetrisched SHARED ${TETRISCHED_SOURCE})
target_include_directories(tetrisched PRIVATE include)
target_compile_features(tetrisched PRIVATE cxx_std_17)

# Include and link all solver backends.
target_include_directories(tetrisched PRIVATE ${SOLVER_BACKEND_INCLUDE_DIRS})
target_link_directories(tetrisched PRIVATE ${SOLVER_BACKEND_LINK_DIRS})
target_link_libraries(tetrisched PRIVATE ${SOLVER_BACKEND_LINK_LIBRARIES})

# Define the target for tests.
set(TETRISCHED_TESTS
     "test/test_expression.cpp"
     "test/test_partitions.cpp"
     "test/test_solver.cpp")

add_executable(test_tetrisched ${TETRISCHED_TESTS})
target_include_directories(test_tetrisched PRIVATE include)

# Include and link solver backends.
target_include_directories(test_tetrisched PRIVATE ${SOLVER_BACKEND_INCLUDE_DIRS})
target_link_directories(test_tetrisched PRIVATE ${SOLVER_BACKEND_LINK_DIRS})

target_link_libraries(
  test_tetrisched
  PRIVATE
  GTest::gtest_main
  tetrisched
  ${SOLVER_BACKEND_LINK_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(test_tetrisched)

# Compilation for the Python bindings.
# Define a Python frontend tetrisched_py and add the relevant sources.
project(tetrisched_py LANGUAGES CXX)
add_subdirectory(extern/pybind11)
pybind11_add_module(
  tetrisched_py
  MODULE
  python/TetrischedPy.cpp
)

# Include the headers from Tetrisched.
target_include_directories(tetrisched_py PRIVATE include)

# Include and link solver backends.
target_include_directories(tetrisched_py PRIVATE ${SOLVER_BACKEND_INCLUDE_DIRS})
target_link_directories(tetrisched_py PRIVATE ${SOLVER_BACKEND_LINK_DIRS})

# Link all the relevant libraries.
target_link_libraries(
  tetrisched_py
  PUBLIC
  tetrisched
)

# Generate an install Python target.
find_package(Python COMPONENTS Interpreter Development)

# Print the found Python path
if (DEFINED Python_SITELIB)
     message(STATUS "Installing Python bindings to ${Python_SITELIB}")
     install(TARGETS tetrisched LIBRARY DESTINATION ${Python_SITELIB})
     install(TARGETS tetrisched tetrisched_py COMPONENT python LIBRARY DESTINATION ${Python_SITELIB})
else()
     message("Cannot install the Python bindings. No suitable Python installation found.")
endif()
