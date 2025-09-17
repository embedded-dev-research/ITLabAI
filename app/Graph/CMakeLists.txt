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
    "https://storage.googleapis.com/kagglesdsdata/datasets/1513816/2500032/test_224/10008.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250916T192850Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=90e54a1e36e1b1be1cda07bcd00eb4cdcf504358bf9ce4eccdf0dc6af6adb19ab9fa82689878a3b26cea4e4295501fdba76e8e5dff3ee0aefe8220abd67ced9667d6f4538a7617bbe4e762a6f97907cab112949353f50276d1911c71dab11ce56370694756a2db16f08c8f819c2dbc8e6c11b131f08481962abfad3347a3ff94469310eb22db163b9036b81ce5efc720b2e175e9bb84beb87e849c2158830697328daa344f03f852ab7dad15c3bc13743f8f185dcfffc9898b7ee449800a188b1809d62f9caeb7343a94c24e7b0cae50abb93cd99a2ee679706eccd5cc093c5f4a9d0f096dcbe76be2c891f75541e11d28f47931cb8bef2dc2fea40ce1ffb391"
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
