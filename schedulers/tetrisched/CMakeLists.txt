cmake_minimum_required(VERSION 3.5...3.26)
project(tetrisched LANGUAGES CXX)

# Compile with atleast C++20.
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
     "src/CPLEXSolver.cpp")


add_library(tetrisched SHARED ${TETRISCHED_SOURCE})
target_include_directories(tetrisched PRIVATE include)
target_compile_features(tetrisched PRIVATE cxx_std_17)

# Include and link CPLEX.
target_include_directories(tetrisched PRIVATE ${CPLEX_INCLUDE_DIRS})
target_link_directories(tetrisched PRIVATE ${CPLEX_LINK_DIRS})
target_link_libraries(tetrisched PRIVATE ${CPLEX_LINK_LIBRARIES})


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

# Include and link CPLEX.
target_include_directories(test_tetrisched PRIVATE ${CPLEX_INCLUDE_DIRS})
target_link_directories(test_tetrisched PRIVATE ${CPLEX_LINK_DIRS})

target_link_libraries(
  test_tetrisched
  PRIVATE
  GTest::gtest_main
  tetrisched
  ${CPLEX_LINK_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(test_tetrisched)
