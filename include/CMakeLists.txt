file(GLOB_RECURSE graph_headers *.h *.hpp)
set(GRAPH_HEADERS "${graph_headers}" PARENT_SCOPE)

file(GLOB_RECURSE layers_headers *.h *.hpp)
set(LAYERS_HEADERS "${layers_headers}" PARENT_SCOPE)

file(GLOB_RECURSE perf_headers *.h *.hpp)
set(PERF_HEADERS "${perf_headers}" PARENT_SCOPE)
