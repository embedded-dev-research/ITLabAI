file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(run_test_read ${TEST_SRC_FILES})
target_link_libraries(run_test_read gtest_main ReadLib)

add_custom_command(TARGET run_test_read POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/Debug/"
        "${CMAKE_BINARY_DIR}/test/test_read/Debug/")

add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test_read)
