add_library(layers_lib STATIC)
target_sources(layers_lib
    PRIVATE
        BatchNormalizationLayer.cpp
        BinaryOpLayer.cpp
        ConcatLayer.cpp
        ConvLayer.cpp
        DropOutLayer.cpp
        EWLayer.cpp
        FCLayer.cpp
        FlattenLayer.cpp
        Layer.cpp
        MatmulLayer.cpp
        OutputLayer.cpp
        PoolingLayer.cpp
        ReduceLayer.cpp
        ReshapeLayer.cpp
        Shape.cpp
        SoftmaxLayer.cpp
        SplitLayer.cpp
        Tensor.cpp
        TransposeLayer.cpp
        ${LAYERS_HEADERS}
)
target_include_directories(layers_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(layers_lib PUBLIC dnnl)
itlabai_link_externals(layers_lib tbb openmp kokkos)
itlabai_target_defaults(layers_lib)
