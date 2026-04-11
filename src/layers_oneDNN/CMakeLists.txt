file(GLOB_RECURSE layers_oneDNN_src *.cpp)
add_library(layers_oneDNN_lib STATIC "${LAYERS_ONEDNN_HEADERS}" "${layers_oneDNN_src}")
target_link_libraries(layers_oneDNN_lib PUBLIC dnnl TBB_unified)
target_include_directories(layers_oneDNN_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_definitions(layers_oneDNN_lib PRIVATE 
    DNNL_ENABLE_CONCURRENT_EXEC=1
)
