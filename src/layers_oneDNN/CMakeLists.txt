file(GLOB LAYERS_ONEDNN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_layers_onednn_lib STATIC)
target_sources(itlabai_layers_onednn_lib PRIVATE ${LAYERS_ONEDNN_SOURCES})
target_include_directories(itlabai_layers_onednn_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_layers_onednn_lib PUBLIC dnnl)
target_link_libraries(itlabai_layers_onednn_lib PUBLIC TBB::tbb Kokkos_imported)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(itlabai_layers_onednn_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(itlabai_layers_onednn_lib PRIVATE 
    DNNL_ENABLE_CONCURRENT_EXEC=1
)
itlabai_target_defaults(itlabai_layers_onednn_lib)
