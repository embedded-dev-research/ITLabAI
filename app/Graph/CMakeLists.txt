set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/build.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/build.cpp")
add_library(BuildGraph STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(BuildGraph PROPERTIES LINKER_LANGUAGE CXX)

find_package(OpenCV REQUIRED PATHS "${OPENCV_BUILD_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(BuildGraph PUBLIC ${OpenCV_LIBS})
target_link_libraries(BuildGraph PUBLIC reader_lib)
target_link_libraries(BuildGraph PUBLIC TBB_unified)
target_link_libraries(BuildGraph PUBLIC layers_lib)
target_link_libraries(BuildGraph PUBLIC gtest_main)

target_include_directories(BuildGraph PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/Json/include)

add_custom_target(create_graph_dirs ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/docs/jsons
    COMMENT "Creating necessary directories"
)

add_executable(Graph_Build graph_build.cpp)
add_executable(ACC_MNIST acc_check_mnist.cpp)

target_link_libraries(Graph_Build BuildGraph)
target_link_libraries(ACC_MNIST BuildGraph)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if (WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
        set(CMAKE_BUILD_TYPE "Release")
    endif()
    add_custom_command(TARGET Graph_Build POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OPENCV_BUILD_DIR}/bin/${CMAKE_BUILD_TYPE}/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

if (WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
        set(CMAKE_BUILD_TYPE "Release")
    endif()
    add_custom_command(TARGET ACC_MNIST POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OPENCV_BUILD_DIR}/bin/${CMAKE_BUILD_TYPE}/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

file(DOWNLOAD
    "https://raw.githubusercontent.com/DeepTrackAI/MNIST_dataset/main/mnist/test/1_000008.png"
    "${CMAKE_CURRENT_BINARY_DIR}/bin/test1.png"
    SHOW_PROGRESS
)

add_definitions(-DIMAGE1_PATH="${CMAKE_CURRENT_BINARY_DIR}/bin/")
add_definitions(-DMODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json")
add_definitions(-DMODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json")
add_definitions(-DMODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json")
add_definitions(-DMNIST_PATH="${CMAKE_SOURCE_DIR}/docs/mnist/mnist/test")