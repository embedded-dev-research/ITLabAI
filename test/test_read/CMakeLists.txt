file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(run_test_read ${TEST_SRC_FILES})
target_link_libraries(run_test_read gtest_main ReadLib)

if (WIN32)
add_custom_command(TARGET run_test_read POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/test/test_read/${CMAKE_BUILD_TYPE}/")
endif()

add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test_read)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
