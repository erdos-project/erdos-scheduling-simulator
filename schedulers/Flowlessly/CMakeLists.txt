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
