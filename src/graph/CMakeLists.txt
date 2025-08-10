file(GLOB_RECURSE graph_src *.cpp)
add_library(graph_lib STATIC "${GRAPH_HEADERS}" "${graph_src}")
target_link_libraries(graph_lib PUBLIC TBB_unified)
