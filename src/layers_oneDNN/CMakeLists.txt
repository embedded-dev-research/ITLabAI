file(GLOB LAYERS_ONEDNN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(layers_oneDNN_lib STATIC)
target_sources(layers_oneDNN_lib PRIVATE ${LAYERS_ONEDNN_SOURCES} ${LAYERS_ONEDNN_HEADERS})
target_include_directories(layers_oneDNN_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(layers_oneDNN_lib PUBLIC dnnl)
itlabai_link_externals(layers_oneDNN_lib tbb kokkos openmp)
target_compile_definitions(layers_oneDNN_lib PRIVATE 
    DNNL_ENABLE_CONCURRENT_EXEC=1
)
itlabai_target_defaults(layers_oneDNN_lib)
