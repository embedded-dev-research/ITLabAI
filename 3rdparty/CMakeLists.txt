add_subdirectory(googletest)

set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
set(DNNL_BUILD_TESTS OFF CACHE BOOL "Build tests") 
set(DNNL_CPU_RUNTIME "OMP" CACHE STRING "CPU runtime")
set(DNNL_GRAPH_DISABLE ON CACHE BOOL "Disable dnnl graph")
set(DNNL_GRAPH_BUILD_TESTS OFF)
set(DNNL_GRAPH_BUILD_EXAMPLES OFF)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "Enable primitive cache")

set(DNNL_ENABLE_CONCURRENT_EXEC OFF CACHE BOOL "Disable concurrent execution")
set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "Disable ITT tasks")
set(DNNL_ENABLE_JIT_DUMP OFF CACHE BOOL "Disable JIT dump")
set(DNNL_ENABLE_JIT_PROFILING OFF CACHE BOOL "Disable JIT profiling")
set(DNNL_ENABLE_MEM_DEBUG OFF CACHE BOOL "Disable memory debug")
set(DNNL_ENABLE_PRIMITIVE_LOG OFF CACHE BOOL "Disable primitive log")
set(DNNL_VERBOSE OFF CACHE BOOL "Disable verbose output")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(DNNL_TARGET_ARCH "AARCH64" CACHE STRING "Target architecture")
    set(DNNL_AARCH64_USE_ACL OFF CACHE BOOL "Disable ARM Compute Library")
    set(DNNL_AARCH64_USE_DNNL OFF CACHE BOOL "Use minimal DNNL for ARM")
endif()

# Для Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DNNL_ARCH_OPT_FLAGS "-O2" CACHE STRING "Architecture optimization flags")
endif()

add_subdirectory(oneDNN)

# Unified TBB Configuration
option(TBB_TEST "Build TBB tests" OFF)
option(TBB_EXAMPLES "Build TBB examples" OFF)
set(TBB_STRICT OFF CACHE BOOL "Treat compiler warnings as errors")

# Configure TBB with unified settings
add_subdirectory(TBB)

# Create a unified TBB interface target
if(NOT TARGET TBB_unified)
    add_library(TBB_unified INTERFACE)
    target_link_libraries(TBB_unified INTERFACE TBB::tbb)
    target_include_directories(TBB_unified INTERFACE 
        $<TARGET_PROPERTY:TBB::tbb,INTERFACE_INCLUDE_DIRECTORIES>
    )
    
    # Platform-specific runtime library handling for Windows
    if(WIN32)
        # Get the TBB library directory based on build type
        if(CMAKE_BUILD_TYPE)
            string(TOLOWER ${CMAKE_BUILD_TYPE} lower_build)
            set(TBB_LIB_DIR "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build/tbb_${lower_build}")
        else()
            set(TBB_LIB_DIR "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build/tbb_debug")
        endif()
        
        # Create a custom target to copy TBB runtime libraries
        add_custom_target(copy_tbb_runtime ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:TBB::tbb>"
                "${CMAKE_BINARY_DIR}/bin/"
            COMMENT "Copying TBB runtime libraries"
        )
        add_dependencies(TBB_unified copy_tbb_runtime)
    endif()
endif()
