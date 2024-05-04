if(TENSORFLOW_FOUND)
    file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
else()
    file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    list(REMOVE_ITEM TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model_read/model_read.cpp")
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
if(TENSORFLOW_FOUND)
    target_link_libraries(run_test PUBLIC AlexNetLib)
endif()

target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")
if(TENSORFLOW_FOUND)
    target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/AlexNet")
endif()

if (WIN32)
add_custom_command(TARGET run_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/test/${CMAKE_BUILD_TYPE}/")
    if(TENSORFLOW_FOUND)
        add_custom_command(TARGET run_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
                "$<TARGET_FILE_DIR:run_test>")
    endif()
endif()

add_test(UnitTests ${CMAKE_BINARY_DIR}/bin/run_test)
set_property(TEST UnitTests PROPERTY ENVIRONMENT_MODIFICATION PATH=path_list_append:${CMAKE_BINARY_DIR}/bin)
set_property(TEST UnitTests PROPERTY ENVIRONMENT_MODIFICATION PATH=path_list_append:${CMAKE_BINARY_DIR}/lib)
set_property(TEST UnitTests PROPERTY ENVIRONMENT_MODIFICATION PATH=path_list_append:${CMAKE_BINARY_DIR}/test/${CMAKE_BUILD_TYPE}/)
set_property(TEST UnitTests PROPERTY ENVIRONMENT_MODIFICATION PATH=path_list_append:$<TARGET_FILE_DIR:run_test>)

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

file(DOWNLOAD
    "https://tinyurl.com/2w6eh53v"
    "${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

target_compile_definitions(run_test PUBLIC TESTS_BINARY_PATH="${CMAKE_CURRENT_BINARY_DIR}")
add_definitions(-DMODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite")
