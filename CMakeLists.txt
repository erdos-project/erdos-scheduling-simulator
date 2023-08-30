cmake_minimum_required(VERSION 3.5...3.26)
project(cpp_schedulers LANGUAGES CXX)
include(GNUInstallDirs)
find_package (Python3 COMPONENTS Interpreter Development)

add_subdirectory(extern/pybind11)
pybind11_add_module(erdos_scheduler_cpp schedulers/erdos_scheduler.cpp)

install(TARGETS erdos_scheduler_cpp
    COMPONENT python
    LIBRARY DESTINATION "${Python3_SITELIB}")
