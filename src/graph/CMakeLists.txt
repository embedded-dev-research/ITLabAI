file(GLOB GRAPH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_graph_lib STATIC)
add_library(ITLabAI::GraphLib ALIAS itlabai_graph_lib)
target_sources(itlabai_graph_lib PRIVATE ${GRAPH_SOURCES})
target_include_directories(itlabai_graph_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
itlabai_link_externals(itlabai_graph_lib tbb kokkos openmp)
itlabai_target_defaults(itlabai_graph_lib)
