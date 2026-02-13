# Common helper functions and defaults for project targets.
#
# Keep this file lightweight: it should not implement a "mini framework".

function(itlabai_target_defaults target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive- /EHsc)
        if(ITLABAI_WERROR)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
        if(ITLABAI_WERROR)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()

    target_compile_features(${target_name} PRIVATE cxx_std_20)

    # Apply project feature defines to every in-tree target.
    if(ITLABAI_FEATURE_DEFS)
        get_target_property(_tgt_type ${target_name} TYPE)
        if(_tgt_type STREQUAL "INTERFACE_LIBRARY")
            target_compile_definitions(${target_name} INTERFACE ${ITLABAI_FEATURE_DEFS})
        else()
            target_compile_definitions(${target_name} PUBLIC ${ITLABAI_FEATURE_DEFS})
        endif()
    endif()
endfunction()

function(itlabai_apply_runtime_rpath target_name)
    set(_paths "")
    foreach(_var IN ITEMS ONEDNN_INSTALL_DIR OPENCV_INSTALL_DIR TBB_INSTALL_DIR KOKKOS_INSTALL_DIR)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _paths "${${_var}}/lib")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _paths)
    if(_paths)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${_paths}"
            SKIP_BUILD_RPATH FALSE
            BUILD_WITH_INSTALL_RPATH FALSE
            INSTALL_RPATH ""
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )
    endif()
endfunction()

