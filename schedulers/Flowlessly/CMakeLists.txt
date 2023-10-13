#####################################################################
project(Flowlessly)
#####################################################################

cmake_minimum_required(VERSION 2.8)

# Path variable configuration
include(cmake/Paths.cmake)

include_directories(${Flowlessly_SOURCE_DIR})
include_directories(${Flowlessly_BUILD_DIR/src})

# CMake libraries
include(CheckLibraryExists)
include(CheckFunctionExists)
include(ExternalProject)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Build options
include(BuildOptions)

# Compiler config
include(CXXCompilerConfig)

# External dependencies
include(ExternalDependencies)

# Doxy documentaion target
include(DoxyTargets)

enable_testing()

subdirs(src)

# Compilation for the Python bindings.
# Define a Python frontend tetrisched_py and add the relevant sources.
project(flowlessly_py LANGUAGES CXX)
add_subdirectory(extern/pybind11)
set(PYBIND_SRC
    src/python/Flowlessly.cpp
)

pybind11_add_module(flowlessly_py ${PYBIND_SRC}
    $<TARGET_OBJECTS:misc>
    $<TARGET_OBJECTS:graphs>
    $<TARGET_OBJECTS:solvers>
)
target_link_libraries(flowlessly_py PRIVATE gflags glog boost_chrono
    boost_system boost_thread boost_timer)

