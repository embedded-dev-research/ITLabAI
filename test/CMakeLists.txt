set(EXCLUDE_FILES "test_read/test_read.cpp" "model_read/model_read.cpp")

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if (NOT TENSORFLOW_FOUND)
    list(REMOVE_ITEM TEST_SRC_FILES ${EXCLUDE_FILES})
endif()

if (NOT WIN32)
    find_package(OpenMP REQUIRED)
endif()

add_executable(run_test ${TEST_SRC_FILES})
if (NOT WIN32)
    target_link_libraries(run_test PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(run_test PUBLIC perf_lib layers_lib)
target_link_libraries(run_test PUBLIC gtest_main)
target_link_libraries(run_test PUBLIC ReadLib)
if (NOT TENSORFLOW_FOUND)
    target_link_libraries(run_test PUBLIC AlexNetLib)
endif()


target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")
target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/AlexNet")

if (WIN32)
add_custom_command(TARGET run_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/test/${CMAKE_BUILD_TYPE}/")
add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
            "$<TARGET_FILE_DIR:run_test>")
endif()

add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DIMAGE_PATH="${CMAKE_CURRENT_BINARY_DIR}/image.jpg")
file(DOWNLOAD
    "https://tinyurl.com/2w6eh53v"
    "${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DMODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite")
