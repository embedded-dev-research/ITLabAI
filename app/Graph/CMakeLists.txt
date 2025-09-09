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

add_executable(ACC_MNIST acc_check_mnist.cpp)
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
    "${CMAKE_SOURCE_DIR}/docs/input/28/test1.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

file(DOWNLOAD
    "https://storage.googleapis.com/kagglesdsdata/datasets/1786300/2914256/train_transformed/cat105.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250907%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250907T025849Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7737d98afd306fe9bd06a3838d444f38407a5d7dc1c2e4cebb747a21cfcd170888b1e3da8ee0bc4d95b5d84e7d6536c934d7945457d4154d228f6f9b31fd76b9b0db6e6bd4d538125c44892bb0c7dd1aae0a495c9e8f69416e2ff9101956abfedadb685d6e7504c575b5e9f5a2465cf8eda7706cde2d7ff0c97a73cc75c06263f1ade60ff392e51b17902b81258a0556f0eaac4c9e175bc0cc16fbeb9bd8df4060f8608e259ecf0d3fa2a33ebb18cd7a7b2f5926bd107c6d1f8c97babb56aa8e59a6d89b78cb56bc9f6202a34ac9434e0f0225502c739d8c9d67dccda7b51acb17212be8de3613e5582e4c20c9d5bc620d5f34733db7ee3511f23ee034f2a1c5"
    "${CMAKE_SOURCE_DIR}/docs/input/224/test1.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

file(DOWNLOAD
    "https://cs13.pikabu.ru/avatars/3329/x3329282-693120225.png"
    "${CMAKE_SOURCE_DIR}/docs/input/256/test1.png"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

add_definitions(-DIMAGE28_PATH="${CMAKE_SOURCE_DIR}/docs/input/28/")
add_definitions(-DIMAGE224_PATH="${CMAKE_SOURCE_DIR}/docs/input/224/")
add_definitions(-DIMAGE256_PATH="${CMAKE_SOURCE_DIR}/docs/input/256/")
add_definitions(-DMODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json")
add_definitions(-DMODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json")
add_definitions(-DMODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json")
add_definitions(-DMODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json")
add_definitions(-DMNIST_PATH="${CMAKE_SOURCE_DIR}/docs/mnist/mnist/test")
