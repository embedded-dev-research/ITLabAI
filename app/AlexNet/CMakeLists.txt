add_executable(Reader_weights reader_weights_sample.cpp)

target_link_libraries(Reader_weights PUBLIC perf_lib layers_lib reader_lib)

add_definitions(-DMODEL_PATH="${CMAKE_SOURCE_DIR}/docs/model_data_alexnet_1.json")

