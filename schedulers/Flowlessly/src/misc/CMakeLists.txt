file(MAKE_DIRECTORY ${Flowlessly_BINARY_DIR}/src/misc)

set(MISC_SRCS
  misc/statistics.cc
  misc/utils.cc
  )

set(MISC_TESTS
  misc/utils_tests.cc
  )

#####################################################################
# Build tests
foreach(T IN ITEMS ${BASE_TESTS})
  get_filename_component(TEST_NAME ${T} NAME_WE)
  add_executable(${TEST_NAME} ${T}
    $<TARGET_OBJECTS:misc>)
  target_link_libraries(${TEST_NAME}
    ${gtest_LIBRARY} ${gtest_MAIN_LIBRARY} glog gflags pthread)
  add_test(${TEST_NAME} ${TEST_NAME})
endforeach(T)