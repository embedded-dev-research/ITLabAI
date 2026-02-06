file(GLOB LAYERS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(layers_lib STATIC)
target_sources(layers_lib PRIVATE ${LAYERS_SOURCES} ${LAYERS_HEADERS})
target_include_directories(layers_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(layers_lib PUBLIC dnnl)
itlabai_link_externals(layers_lib tbb openmp kokkos)
itlabai_target_defaults(layers_lib)
