file(MAKE_DIRECTORY ${Flowlessly_BINARY_DIR}/src/solvers)

set(SOLVERS_SRCS
  solvers/cycle_cancelling.cc
  solvers/successive_shortest.cc
  )

set(SOLVERS_TESTS
  solvers/cycle_cancelling_tests.cc
  solvers/successive_shortest_tests.cc
)

#####################################################################
# Build tests
foreach(T IN ITEMS ${BASE_TESTS})
  get_filename_component(TEST_NAME ${T} NAME_WE)
  add_executable(${TEST_NAME} ${T}
    $<TARGET_OBJECTS:solvers>)
  target_link_libraries(${TEST_NAME}
    ${gtest_LIBRARY} ${gtest_MAIN_LIBRARY} glog gflags pthread)
  add_test(${TEST_NAME} ${TEST_NAME})
endforeach(T)