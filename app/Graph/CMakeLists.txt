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
    "https://storage.googleapis.com/kagglesdsdata/datasets/1513816/2500032/test_224/10008.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250911T144346Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=07e6d4f5d018e5858d046fc697dbcc726a423cb5c9eff15a6ff973a13060bf9300b1395cb641966de90d11ebf59d7b8650c8c68121bf7e447de375526ab4586b0906db71d5623bee96a9d4e289d15165e3c2b08e04928328f8540b03cb77585082e2acc9be5c61ebc51a08c8b010ba0b6f1192344b6828d3b935dde195ecdca77476483abe7784df5f569b7bd1e4e29b1c670b9f35b76a7e4a9dc6b2b0705654753e81a91a579c0c071338aa215917f29f9ee84c9bef9c805254c917347b2c3a9a31501c1238be23296009d6617f74c1070294f6b56ac0314ea7162a6adddfb9306c5333bd879a24796511261084dd2d2dbbc515c7917ebd91aac735359d1789"
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
