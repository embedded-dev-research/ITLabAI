add_subdirectory(googletest)

set(DNNL_CPU_RUNTIME "TBB" CACHE STRING "oneDNN CPU threading runtime")

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
