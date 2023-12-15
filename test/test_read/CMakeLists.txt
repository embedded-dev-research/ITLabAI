file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(run_test_read ${TEST_SRC_FILES})
target_link_libraries(run_test_read gtest_main MyInclude)
add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test_read)


