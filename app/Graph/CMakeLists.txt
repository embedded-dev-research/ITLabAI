set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/build.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/build.cpp")
add_library(BuildGraph STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(BuildGraph PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(BuildGraph PUBLIC reader_lib layers_lib layers_oneDNN_lib OpenCV::opencv_world Kokkos_imported TBB_unified)
itlabai_link_externals(BuildGraph opencv tbb gtest kokkos openmp)
add_dependencies(BuildGraph opencv_external)
target_compile_definitions(BuildGraph PUBLIC
    IMAGE28_PATH="${CMAKE_SOURCE_DIR}/docs/input/28/"
    IMAGENET_ACC="${CMAKE_SOURCE_DIR}/docs/ImageNet/test/"
    IMAGENET_PATH="${CMAKE_SOURCE_DIR}/docs/input/Imagenet_test/"
    MODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json"
    IMAGENET_LABELS="${CMAKE_SOURCE_DIR}/docs/imagenet1000_clsidx_to_labels.json"
    MNIST_PATH="${CMAKE_SOURCE_DIR}/docs/mnist/mnist/test"
)

add_executable(Graph_Build graph_build.cpp)
target_link_libraries(Graph_Build PRIVATE BuildGraph)
itlabai_target_defaults(Graph_Build)
itlabai_apply_runtime_rpath(Graph_Build)

add_executable(ACC acc_check.cpp)
target_link_libraries(ACC PRIVATE BuildGraph OpenCV::opencv_world)
itlabai_target_defaults(ACC)
itlabai_apply_runtime_rpath(ACC)

add_executable(onnx_subgraphs onnx_subgraphs.cpp)
target_link_libraries(onnx_subgraphs PRIVATE BuildGraph)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(onnx_subgraphs PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(onnx_subgraphs PRIVATE graphT_lib)
itlabai_target_defaults(onnx_subgraphs)
itlabai_apply_runtime_rpath(onnx_subgraphs)
