set(GRAPH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/runtime_options.hpp
)
set(GRAPHT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_transformations/graph_transformations.hpp
)
set(LAYERS_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/BatchNormalizationLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/BinaryOpLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/ConcatLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/ConvLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/DropOutLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/EWLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/FCLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/FlattenLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/InputLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/Layer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/MatmulLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/OutputLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/PoolingLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/ReduceLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/ReshapeLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/Shape.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/SoftmaxLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/SplitLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/Tensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/TransposeLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parallel/backends.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parallel/parallel.hpp
)
set(LAYERS_ONEDNN_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/layers_oneDNN/ConvLayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layers_oneDNN/EWLayer.hpp
)
set(PERF_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/perf/benchmarking.hpp
)
set(READER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Weights_Reader/reader_weights.hpp
)
set(GRAPH_HEADERS "${GRAPH_HEADERS}" PARENT_SCOPE)
set(GRAPHT_HEADERS "${GRAPHT_HEADERS}" PARENT_SCOPE)
set(LAYERS_HEADERS "${LAYERS_HEADERS}" PARENT_SCOPE)
set(LAYERS_ONEDNN_HEADERS "${LAYERS_ONEDNN_HEADERS}" PARENT_SCOPE)
set(PERF_HEADERS "${PERF_HEADERS}" PARENT_SCOPE)
set(READER_HEADERS "${READER_HEADERS}" PARENT_SCOPE)
