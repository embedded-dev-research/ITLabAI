# Lightweight FindTBB to reuse vendored TBB target from this project

if(TARGET TBB::tbb)
    set(_vendor_tbb_include "${CMAKE_SOURCE_DIR}/3rdparty/TBB/include")
    if(EXISTS "${_vendor_tbb_include}/oneapi/tbb/version.h")
        # TBB::tbb is an ALIAS; set properties on the real target 'tbb'
        if(TARGET tbb)
            set_target_properties(tbb PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_vendor_tbb_include}")
        endif()
    endif()
    set(TBB_FOUND TRUE)
    # Provide minimal compatibility variables
    set(TBB_IMPORTED_TARGETS TBB::tbb)
    mark_as_advanced(TBB_FOUND)
    unset(_vendor_tbb_include)
    return()
endif()

# Create an imported INTERFACE target that links to our vendored TBB build target
add_library(TBB::tbb INTERFACE IMPORTED)
set_target_properties(TBB::tbb PROPERTIES
    INTERFACE_LINK_LIBRARIES tbb
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/3rdparty/TBB/include"
)
set(TBB_FOUND TRUE)
set(TBB_IMPORTED_TARGETS TBB::tbb)
