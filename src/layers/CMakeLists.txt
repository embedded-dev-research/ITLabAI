file(GLOB_RECURSE layers_src *.cpp)
add_library(layers_lib STATIC "${LAYERS_HEADERS}" "${layers_src}")

target_link_libraries(layers_lib PUBLIC TBB_unified)
target_link_libraries(layers_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(layers_lib PUBLIC dnnl)
target_link_libraries(layers_lib PUBLIC Kokkos_imported)

if(TARGET AdaptiveCpp::acpp-rt)
    target_compile_definitions(layers_lib PUBLIC ITLABAI_HAS_SYCL)
    if(WIN32)
        target_compile_definitions(layers_lib PUBLIC
            _ENABLE_EXTENDED_ALIGNED_STORAGE
            _USE_MATH_DEFINES
            __ACPP_ENABLE_OMPHOST_TARGET__
        )
    endif()
    target_link_libraries(layers_lib PUBLIC AdaptiveCpp::acpp-rt)
    if(NOT WIN32)
        add_sycl_to_target(TARGET layers_lib SOURCES ${layers_src})
    endif()
endif()
