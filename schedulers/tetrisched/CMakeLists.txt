cmake_minimum_required(VERSION 3.5...3.26)
project(tetrisched LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

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
     "src/tetrisched.cpp"
     "src/Expression.cpp"
     "src/Solver.cpp"
     "src/SolverModel.cpp"
     "src/Util.cpp")

add_executable(tetrisched ${TETRISCHED_SOURCE})
target_include_directories(tetrisched PRIVATE ${CPLEX_INCLUDE_DIRS})
target_link_directories(tetrisched PRIVATE ${CPLEX_LINK_DIRS})
target_link_libraries(tetrisched PRIVATE ${CPLEX_LINK_LIBRARIES})
