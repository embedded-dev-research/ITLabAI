file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(run_test ${TEST_SRC_FILES})
target_link_libraries(run_test gtest_main MyInclude)
add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test)


