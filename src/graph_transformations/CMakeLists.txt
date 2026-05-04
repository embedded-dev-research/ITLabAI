file(GLOB_RECURSE graphT_src *.cpp)
add_library(graphT_lib STATIC "${GRAPHT_HEADERS}" "${graphT_src}")
target_link_libraries(graphT_lib PUBLIC graph_lib layers_fused_lib TBB_unified)
