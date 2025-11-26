file(GLOB_RECURSE layers_src *.cpp)
add_library(layers_lib STATIC "${LAYERS_HEADERS}" "${layers_src}")

target_include_directories(layers_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(layers_lib PUBLIC TBB_unified)
target_link_libraries(layers_lib PUBLIC dnnl)
