file(GLOB BUILDGRAPH_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB BUILDGRAPH_HPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
list(FILTER BUILDGRAPH_CPP INCLUDE REGEX "/build\\.cpp$")

add_library(BuildGraph STATIC ${BUILDGRAPH_HPP} ${BUILDGRAPH_CPP})

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

file(GLOB GRAPH_BUILD_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/graph_build.cpp")
add_executable(Graph_Build ${GRAPH_BUILD_MAIN})
target_link_libraries(Graph_Build PRIVATE BuildGraph)
itlabai_target_defaults(Graph_Build)
itlabai_apply_runtime_rpath(Graph_Build)

file(GLOB ACC_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/acc_check.cpp")
add_executable(ACC ${ACC_MAIN})
target_link_libraries(ACC PRIVATE BuildGraph OpenCV::opencv_world)
itlabai_target_defaults(ACC)
itlabai_apply_runtime_rpath(ACC)

file(GLOB ONNX_SUBGRAPHS_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_subgraphs.cpp")
add_executable(onnx_subgraphs ${ONNX_SUBGRAPHS_MAIN})
target_link_libraries(onnx_subgraphs PRIVATE BuildGraph)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(onnx_subgraphs PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(onnx_subgraphs PRIVATE graphT_lib)
itlabai_target_defaults(onnx_subgraphs)
itlabai_apply_runtime_rpath(onnx_subgraphs)
