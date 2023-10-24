file(MAKE_DIRECTORY ${Flowlessly_BINARY_DIR}/src/graphs)

set(GRAPHS_SRCS
  graphs/adjacency_map_graph.cc
  graphs/graph.cc
  graphs/graph_generator.cc
  )

set(GRAPHS_TESTS
  graphs/adjacency_map_graph_tests.cc
)

#####################################################################
# Build tests
foreach(T IN ITEMS ${BASE_TESTS})
  get_filename_component(TEST_NAME ${T} NAME_WE)
  add_executable(${TEST_NAME} ${T}
    $<TARGET_OBJECTS:graphs>)
  target_link_libraries(${TEST_NAME}
    ${gtest_LIBRARY} ${gtest_MAIN_LIBRARY} glog gflags pthread)
  add_test(${TEST_NAME} ${TEST_NAME})
endforeach(T)