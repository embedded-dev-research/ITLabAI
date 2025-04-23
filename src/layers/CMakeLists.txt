file(GLOB_RECURSE layers_src *.cpp)
add_library(layers_lib STATIC "${LAYERS_HEADERS}" "${layers_src}")
target_link_libraries(layers_lib PUBLIC TBB::tbb)
if (WIN32)
    string(TOLOWER ${CMAKE_BUILD_TYPE} lower_build)
    file(GLOB_RECURSE tbb_files "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build/tbb_${lower_build}/tbb*")
    add_custom_command(TARGET layers_lib POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${tbb_files}
                       "${CMAKE_BINARY_DIR}/bin/")
endif()
