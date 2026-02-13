file(GLOB BUILDGRAPH_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB BUILDGRAPH_HPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
list(FILTER BUILDGRAPH_CPP INCLUDE REGEX "/build\\.cpp$")

add_library(itlabai_build_graph_lib STATIC ${BUILDGRAPH_HPP} ${BUILDGRAPH_CPP})

set_target_properties(itlabai_build_graph_lib PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(itlabai_build_graph_lib PUBLIC
    itlabai_reader_lib
    itlabai_layers_lib
    itlabai_layers_onednn_lib
)
target_link_libraries(itlabai_build_graph_lib PUBLIC OpenCV::opencv_world TBB::tbb Kokkos_imported)
target_compile_definitions(itlabai_build_graph_lib PUBLIC
    IMAGE28_PATH="${ITLABAI_TEST_DATA_DIR}/28/"
    IMAGENET_ACC="${ITLABAI_IMAGENET_ACC_DIR}/"
    IMAGENET_PATH="${ITLABAI_TEST_DATA_DIR}/Imagenet_test/"
    MODEL_PATH_H5="${ITLABAI_MODELS_JSON_DIR}/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${ITLABAI_MODELS_JSON_DIR}/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${ITLABAI_MODELS_JSON_DIR}/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${ITLABAI_MODELS_JSON_DIR}/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${ITLABAI_MODELS_JSON_DIR}/yolo11x-cls_onnx_model.json"
    IMAGENET_LABELS="${ITLABAI_IMAGENET_LABELS_FILE}"
    MNIST_PATH="${ITLABAI_MNIST_DIR}"
)

file(GLOB GRAPH_BUILD_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/graph_build.cpp")
add_executable(itlabai_graph_build_exe ${GRAPH_BUILD_MAIN})
set_target_properties(itlabai_graph_build_exe PROPERTIES OUTPUT_NAME "itlabai_graph_build")
target_link_libraries(itlabai_graph_build_exe PRIVATE itlabai_build_graph_lib)
add_dependencies(itlabai_graph_build_exe itlabai_fetch_test_data)
itlabai_target_defaults(itlabai_graph_build_exe)

file(GLOB ACC_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/acc_check.cpp")
add_executable(itlabai_acc_exe ${ACC_MAIN})
set_target_properties(itlabai_acc_exe PROPERTIES OUTPUT_NAME "itlabai_acc")
target_link_libraries(itlabai_acc_exe PRIVATE itlabai_build_graph_lib)
add_dependencies(itlabai_acc_exe itlabai_fetch_test_data)
itlabai_target_defaults(itlabai_acc_exe)

file(GLOB ONNX_SUBGRAPHS_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_subgraphs.cpp")
add_executable(itlabai_onnx_subgraphs_exe ${ONNX_SUBGRAPHS_MAIN})
set_target_properties(itlabai_onnx_subgraphs_exe PROPERTIES OUTPUT_NAME "itlabai_onnx_subgraphs")
target_link_libraries(itlabai_onnx_subgraphs_exe PRIVATE itlabai_build_graph_lib)
add_dependencies(itlabai_onnx_subgraphs_exe itlabai_fetch_test_data)
target_link_libraries(itlabai_onnx_subgraphs_exe PRIVATE itlabai_graph_transformations_lib)
itlabai_target_defaults(itlabai_onnx_subgraphs_exe)
