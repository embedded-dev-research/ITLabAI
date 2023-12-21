cmake_minimum_required(VERSION 3.20)

project(itlab_2023)

enable_testing()

include_directories("include")

add_subdirectory(3rdparty)
add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

add_library(graph_lib STATIC "${GRAPH_HEADERS}" "${GRAPH_SRC}")
add_library(perf_lib STATIC "${PERF_HEADERS}" "${PERF_SRC}")
