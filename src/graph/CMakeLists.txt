file(GLOB GRAPH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_graph_lib STATIC)
target_sources(itlabai_graph_lib PRIVATE ${GRAPH_SOURCES})
target_include_directories(itlabai_graph_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_graph_lib PUBLIC TBB::tbb Kokkos_imported)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(itlabai_graph_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
itlabai_target_defaults(itlabai_graph_lib)
