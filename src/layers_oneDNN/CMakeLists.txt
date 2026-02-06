file(GLOB LAYERS_ONEDNN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_layers_onednn_lib STATIC)
add_library(ITLabAI::LayersOneDnnLib ALIAS itlabai_layers_onednn_lib)
target_sources(itlabai_layers_onednn_lib PRIVATE ${LAYERS_ONEDNN_SOURCES} ${LAYERS_ONEDNN_HEADERS})
target_include_directories(itlabai_layers_onednn_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_layers_onednn_lib PUBLIC dnnl)
itlabai_link_externals(itlabai_layers_onednn_lib tbb kokkos openmp)
target_compile_definitions(itlabai_layers_onednn_lib PRIVATE 
    DNNL_ENABLE_CONCURRENT_EXEC=1
)
itlabai_target_defaults(itlabai_layers_onednn_lib)
