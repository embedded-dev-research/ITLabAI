file(GLOB_RECURSE graph_headers graph/*.h graph/*.hpp)
set(GRAPH_HEADERS "${graph_headers}" PARENT_SCOPE)

file(GLOB_RECURSE graphT_headers graph_transformations/*.h graph_transformations/*.hpp)
set(GRAPHT_HEADERS "${graphT_headers}" PARENT_SCOPE)

file(GLOB_RECURSE layers_headers layers/*.h layers/*.hpp)
set(LAYERS_HEADERS "${layers_headers}" PARENT_SCOPE)

file(GLOB_RECURSE perf_headers perf/*.h perf/*.hpp)
set(PERF_HEADERS "${perf_headers}" PARENT_SCOPE)

file(GLOB_RECURSE reader_headers Weights_Reader/*.h Weights_Reader/*.hpp)
set(READER_HEADERS "${reader_headers}" PARENT_SCOPE)
