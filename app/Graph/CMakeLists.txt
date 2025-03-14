file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")

execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_apps=OFF
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/build.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/build.cpp")
add_library(BuildGraph STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(BuildGraph PROPERTIES LINKER_LANGUAGE CXX)

find_package(OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(BuildGraph PUBLIC ${OpenCV_LIBS})
target_link_libraries(BuildGraph PUBLIC reader_lib)
target_link_libraries(BuildGraph PUBLIC TBB::tbb)
target_link_libraries(BuildGraph PUBLIC layers_lib)
target_link_libraries(BuildGraph PUBLIC gtest_main)

target_include_directories(BuildGraph PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/Json/include)

add_executable(Graph_Build graph_build.cpp)
target_link_libraries(Graph_Build BuildGraph)

add_executable(ACC_MNIST acc_check_mnist.cpp)
target_link_libraries(ACC_MNIST BuildGraph)

if (WIN32)
add_custom_command(TARGET Graph_Build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/app/ReaderImage/${CMAKE_BUILD_TYPE}/")
endif()

if (WIN32)
add_custom_command(TARGET ACC_MNIST POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/app/ReaderImage/${CMAKE_BUILD_TYPE}/")
endif()

file(DOWNLOAD
    "https://raw.githubusercontent.com/DeepTrackAI/MNIST_dataset/main/mnist/test/1_000008.png"
    "${CMAKE_SOURCE_DIR}/docs/input/test1.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

add_definitions(-DIMAGE1_PATH="${CMAKE_SOURCE_DIR}/docs/input/")
add_definitions(-DMODEL_PATH="${CMAKE_SOURCE_DIR}/docs/model_data_alexnet_1.json")
add_definitions(-DMNIST_PATH="${CMAKE_SOURCE_DIR}/docs/mnist/mnist/test")
