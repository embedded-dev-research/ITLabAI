file(GLOB READER_WEIGHTS_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_weights_sample.cpp")
file(GLOB READER_WEIGHTS_ONNX_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_weights_sample_onnx.cpp")
add_executable(itlabai_reader_weights_exe ${READER_WEIGHTS_SRC})
add_executable(ITLabAI::ReaderWeightsExe ALIAS itlabai_reader_weights_exe)
add_executable(itlabai_reader_weights_onnx_exe ${READER_WEIGHTS_ONNX_SRC})
add_executable(ITLabAI::ReaderWeightsOnnxExe ALIAS itlabai_reader_weights_onnx_exe)
target_link_libraries(itlabai_reader_weights_exe PUBLIC itlabai_perf_lib itlabai_layers_lib itlabai_reader_lib)
target_link_libraries(itlabai_reader_weights_onnx_exe PUBLIC itlabai_perf_lib itlabai_layers_lib itlabai_reader_lib)
itlabai_target_defaults(itlabai_reader_weights_exe)
itlabai_target_defaults(itlabai_reader_weights_onnx_exe)
itlabai_apply_runtime_rpath(itlabai_reader_weights_exe)
itlabai_apply_runtime_rpath(itlabai_reader_weights_onnx_exe)
target_compile_definitions(itlabai_reader_weights_exe PRIVATE
    MODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json"
)
target_compile_definitions(itlabai_reader_weights_onnx_exe PRIVATE
    MODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json"
)
