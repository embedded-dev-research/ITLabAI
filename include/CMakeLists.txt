file(GLOB_RECURSE graph_headers graph/*.h graph/*.hpp)
set(GRAPH_HEADERS "${graph_headers}" PARENT_SCOPE)

file(GLOB_RECURSE layers_headers layers/*.h layers/*.hpp)
set(LAYERS_HEADERS "${layers_headers}" PARENT_SCOPE)

file(GLOB_RECURSE perf_headers perf/*.h perf/*.hpp)
set(PERF_HEADERS "${perf_headers}" PARENT_SCOPE)

file(GLOB_RECURSE alexnet_headers alexnet/*.h alexnet/*.hpp)
set(ALEXNET_HEADERS "${alexnet_headers}" PARENT_SCOPE)
