add_executable(Reader_weights reader_weights_sample.cpp)
add_executable(Reader_weights_onnx reader_weights_sample_onnx.cpp)
target_link_libraries(Reader_weights PUBLIC perf_lib layers_lib reader_lib)
target_link_libraries(Reader_weights_onnx PUBLIC perf_lib layers_lib reader_lib)
itlabai_target_defaults(Reader_weights)
itlabai_target_defaults(Reader_weights_onnx)
itlabai_apply_runtime_rpath(Reader_weights)
itlabai_apply_runtime_rpath(Reader_weights_onnx)
target_compile_definitions(Reader_weights PRIVATE
    MODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json"
)
target_compile_definitions(Reader_weights_onnx PRIVATE
    MODEL_PATH_H5="${CMAKE_SOURCE_DIR}/docs/jsons/model_data_alexnet_1.json"
    MODEL_PATH_GOOGLENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/googlenet_onnx_model.json"
    MODEL_PATH_DENSENET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/densenet121_Opset16_onnx_model.json"
    MODEL_PATH_RESNET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/resnest101e_Opset16_onnx_model.json"
    MODEL_PATH_YOLO11NET_ONNX="${CMAKE_SOURCE_DIR}/docs/jsons/yolo11x-cls_onnx_model.json"
)
