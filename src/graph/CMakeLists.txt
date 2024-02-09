file(GLOB_RECURSE graph_src *.cpp)
add_library(graph_lib STATIC "${GRAPH_HEADERS}" "${graph_src}")
