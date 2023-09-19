cmake_minimum_required(VERSION 3.5...3.26)
project(tetrisched LANGUAGES CXX)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(CPLEX_INCLUDE_DIRS
     "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/include/"
     "/opt/ibm/ILOG/CPLEX_Studio2211/concert/include/")
set(CPLEX_LINK_DIRS
     "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/lib/x86-64_linux/static_pic/"
     "/opt/ibm/ILOG/CPLEX_Studio2211/concert/lib/x86-64_linux/static_pic/")
set(CPLEX_LINK_LIBRARIES
     "ilocplex"
     "concert"
     "cplex"
     "m"
     "pthread"
     "dl")

set(TETRISCHED_SOURCE
     "src/Expression.cpp"
     "src/Worker.cpp"
     "src/Partition.cpp"
     "src/SolverModel.cpp"
     "src/Solver.cpp")


add_library(tetrisched SHARED ${TETRISCHED_SOURCE})
target_include_directories(tetrisched PRIVATE include)
target_compile_features(tetrisched PRIVATE cxx_std_17)

# add_executable(tetrisched_main ${TETRISCHED_SOURCE})
# target_include_directories(tetrisched_main PRIVATE ${CPLEX_INCLUDE_DIRS})
# target_link_directories(tetrisched_main PRIVATE ${CPLEX_LINK_DIRS})
# target_link_libraries(tetrisched_main PRIVATE ${CPLEX_LINK_LIBRARIES})
#
# enable_testing()
#

set(TETRISCHED_TESTS
     "test/test_expression.cpp"
     "test/test_partitions.cpp"
     "test/test_solver.cpp"
     "test/test_work.cpp")

add_executable(test_tetrisched ${TETRISCHED_TESTS})
target_include_directories(test_tetrisched PRIVATE include)
target_compile_features(test_tetrisched PRIVATE cxx_std_17)
target_link_libraries(
  test_tetrisched
  GTest::gtest_main
  tetrisched
)

include(GoogleTest)
gtest_discover_tests(test_tetrisched)
