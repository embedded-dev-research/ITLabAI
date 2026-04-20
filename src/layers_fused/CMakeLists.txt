file(GLOB_RECURSE layers_fused_src *.cpp)
add_library(layers_fused_lib STATIC "${LAYERS_FUSED_HEADERS}" "${layers_fused_src}")

target_link_libraries(layers_fused_lib PUBLIC layers_lib)
target_link_libraries(layers_fused_lib PUBLIC TBB_unified)
target_link_libraries(layers_fused_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(layers_fused_lib PUBLIC dnnl)
