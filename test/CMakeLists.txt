file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(run_test ${TEST_SRC_FILES})
if (NOT WIN32)
    target_link_libraries(run_test PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(run_test PUBLIC perf_lib layers_lib)
target_link_libraries(run_test PUBLIC gtest)
target_link_libraries(run_test PUBLIC ReadLib)
target_link_libraries(run_test PUBLIC reader_lib)
target_link_libraries(run_test PUBLIC graph_lib)
target_link_libraries(run_test PUBLIC graphT_lib)


target_include_directories(run_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if (WIN32)
    add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OPENCV_BUILD_DIR}/bin/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

add_test(UnitTests ${CMAKE_BINARY_DIR}/bin/run_test)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

file(DOWNLOAD
    "https://raw.githubusercontent.com/bpinaya/AlexNetRT/master/data/alexnet/imagenet-labels.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/imagenet-labels.txt"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
target_compile_definitions(run_test PRIVATE TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/test/model_read/json_for_test")


target_compile_definitions(run_test PUBLIC TESTS_BINARY_PATH="${CMAKE_CURRENT_BINARY_DIR}")

