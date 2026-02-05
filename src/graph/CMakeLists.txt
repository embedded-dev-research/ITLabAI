add_library(graph_lib STATIC)
target_sources(graph_lib PRIVATE graph.cpp ${GRAPH_HEADERS})
target_include_directories(graph_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
itlabai_link_externals(graph_lib tbb kokkos openmp)
itlabai_target_defaults(graph_lib)
