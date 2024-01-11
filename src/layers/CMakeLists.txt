file(GLOB_RECURSE layers_src *.cpp)
add_library(layers_lib STATIC "${LAYERS_HEADERS}" "${layers_src}")
