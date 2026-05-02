add_executable(SYCL_Example
    sycl_example.cpp
)
target_link_libraries(SYCL_Example PRIVATE layers_lib AdaptiveCpp::acpp-rt)

if(WIN32)
    target_sources(SYCL_Example PRIVATE sycl_kernel.cpp)
    target_compile_definitions(SYCL_Example PRIVATE _USE_MATH_DEFINES ITLABAI_HAS_SYCL)
else()
    add_library(SYCL_Example_kernel OBJECT sycl_kernel.cpp)
    target_compile_definitions(SYCL_Example_kernel PRIVATE ITLABAI_HAS_SYCL)
    target_link_libraries(SYCL_Example_kernel PRIVATE AdaptiveCpp::acpp-rt TBB_unified Kokkos_imported)
    add_sycl_to_target(TARGET SYCL_Example_kernel SOURCES
        sycl_kernel.cpp
    )
    target_sources(SYCL_Example PRIVATE $<TARGET_OBJECTS:SYCL_Example_kernel>)
endif()

add_test(NAME SYCL.Example COMMAND SYCL_Example)
set_tests_properties(SYCL.Example PROPERTIES
    LABELS "sycl;smoke"
    TIMEOUT 60
)
