file(GLOB BUILDGRAPH_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB BUILDGRAPH_HPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
list(FILTER BUILDGRAPH_CPP INCLUDE REGEX "/build\\.cpp$")

add_library(itlabai_build_graph_lib STATIC ${BUILDGRAPH_HPP} ${BUILDGRAPH_CPP})
add_library(ITLabAI::BuildGraphLib ALIAS itlabai_build_graph_lib)

set_target_properties(itlabai_build_graph_lib PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(itlabai_build_graph_lib PUBLIC itlabai_reader_lib itlabai_layers_lib itlabai_layers_onednn_lib OpenCV::opencv_world Kokkos_imported TBB_unified)
itlabai_link_externals(itlabai_build_graph_lib opencv tbb gtest kokkos openmp)
add_dependencies(itlabai_build_graph_lib opencv_external)
target_compile_definitions(itlabai_build_graph_lib PUBLIC
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
add_executable(itlabai_graph_build_exe ${GRAPH_BUILD_MAIN})
add_executable(ITLabAI::GraphBuildExe ALIAS itlabai_graph_build_exe)
target_link_libraries(itlabai_graph_build_exe PRIVATE itlabai_build_graph_lib)
itlabai_target_defaults(itlabai_graph_build_exe)
itlabai_apply_runtime_rpath(itlabai_graph_build_exe)

file(GLOB ACC_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/acc_check.cpp")
add_executable(itlabai_acc_exe ${ACC_MAIN})
add_executable(ITLabAI::AccExe ALIAS itlabai_acc_exe)
target_link_libraries(itlabai_acc_exe PRIVATE itlabai_build_graph_lib OpenCV::opencv_world)
itlabai_target_defaults(itlabai_acc_exe)
itlabai_apply_runtime_rpath(itlabai_acc_exe)

file(GLOB ONNX_SUBGRAPHS_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_subgraphs.cpp")
add_executable(itlabai_onnx_subgraphs_exe ${ONNX_SUBGRAPHS_MAIN})
add_executable(ITLabAI::OnnxSubgraphsExe ALIAS itlabai_onnx_subgraphs_exe)
target_link_libraries(itlabai_onnx_subgraphs_exe PRIVATE itlabai_build_graph_lib)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(itlabai_onnx_subgraphs_exe PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(itlabai_onnx_subgraphs_exe PRIVATE itlabai_graph_transformations_lib)
itlabai_target_defaults(itlabai_onnx_subgraphs_exe)
itlabai_apply_runtime_rpath(itlabai_onnx_subgraphs_exe)
