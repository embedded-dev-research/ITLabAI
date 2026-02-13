file(GLOB LAYERS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_layers_lib STATIC)
target_sources(itlabai_layers_lib PRIVATE ${LAYERS_SOURCES})
target_include_directories(itlabai_layers_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_layers_lib PUBLIC dnnl)
target_link_libraries(itlabai_layers_lib PUBLIC TBB::tbb Kokkos_imported)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(itlabai_layers_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
itlabai_target_defaults(itlabai_layers_lib)
