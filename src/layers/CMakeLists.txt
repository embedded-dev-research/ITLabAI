file(GLOB LAYERS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_layers_lib STATIC)
add_library(ITLabAI::LayersLib ALIAS itlabai_layers_lib)
target_sources(itlabai_layers_lib PRIVATE ${LAYERS_SOURCES} ${LAYERS_HEADERS})
target_include_directories(itlabai_layers_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_layers_lib PUBLIC dnnl)
itlabai_link_externals(itlabai_layers_lib tbb openmp kokkos)
itlabai_target_defaults(itlabai_layers_lib)
