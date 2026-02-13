file(GLOB READER_WEIGHTS_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_weights_sample.cpp")
file(GLOB READER_WEIGHTS_ONNX_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_weights_sample_onnx.cpp")

set(_itlabai_model_defs
    MODEL_PATH_H5="${ITLABAI_MODELS_JSON_DIR}/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${ITLABAI_MODELS_JSON_DIR}/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${ITLABAI_MODELS_JSON_DIR}/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${ITLABAI_MODELS_JSON_DIR}/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${ITLABAI_MODELS_JSON_DIR}/yolo11x-cls_onnx_model.json"
)
add_executable(itlabai_reader_weights_exe ${READER_WEIGHTS_SRC})
add_executable(itlabai_reader_weights_onnx_exe ${READER_WEIGHTS_ONNX_SRC})
set_target_properties(itlabai_reader_weights_exe PROPERTIES OUTPUT_NAME "itlabai_reader_weights")
set_target_properties(itlabai_reader_weights_onnx_exe PROPERTIES OUTPUT_NAME "itlabai_reader_weights_onnx")
target_link_libraries(itlabai_reader_weights_exe PUBLIC itlabai_perf_lib itlabai_layers_lib itlabai_reader_lib)
target_link_libraries(itlabai_reader_weights_onnx_exe PUBLIC itlabai_perf_lib itlabai_layers_lib itlabai_reader_lib)
itlabai_target_defaults(itlabai_reader_weights_exe)
itlabai_target_defaults(itlabai_reader_weights_onnx_exe)
target_compile_definitions(itlabai_reader_weights_exe PRIVATE ${_itlabai_model_defs})
target_compile_definitions(itlabai_reader_weights_onnx_exe PRIVATE ${_itlabai_model_defs})
