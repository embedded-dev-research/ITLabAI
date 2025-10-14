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

add_executable(Graph_Build graph_build.cpp)
target_link_libraries(Graph_Build BuildGraph)

add_executable(ACC acc_check.cpp)
target_link_libraries(ACC BuildGraph)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if (WIN32)
    add_custom_command(TARGET Graph_Build POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OPENCV_BUILD_DIR}/bin/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

if (WIN32)
	if ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	    set(CMAKE_BUILD_TYPE "Debug")
	endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
	    set(CMAKE_BUILD_TYPE "Release")
	endif()
    add_custom_command(TARGET ACC POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OPENCV_BUILD_DIR}/bin/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

file(DOWNLOAD
    "https://raw.githubusercontent.com/DeepTrackAI/MNIST_dataset/main/mnist/test/1_000008.png"
    "${CMAKE_SOURCE_DIR}/docs/input/28/test1.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

file(DOWNLOAD
    "blob:https://ru.pinterest.com/63b88674-b4a6-4ef3-85b2-ab57ef7bb8e7"
    "${CMAKE_SOURCE_DIR}/docs/input/Imagenet_test/tench.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

add_definitions(-DIMAGE28_PATH="${CMAKE_SOURCE_DIR}/docs/input/28/")
add_definitions(-DIMAGENET_ACC="${CMAKE_SOURCE_DIR}/docs/ImageNet/test/")
add_definitions(-DIMAGENET_PATH="${CMAKE_SOURCE_DIR}/docs/input/Imagenet_test/")
add_definitions(-DMODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json")
add_definitions(-DMODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json")
add_definitions(-DMODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json")
add_definitions(-DIMAGENET_LABELS="${CMAKE_SOURCE_DIR}/docs/imagenet1000_clsidx_to_labels.json")
add_definitions(-DMNIST_PATH="${CMAKE_SOURCE_DIR}/docs/mnist/mnist/test")
