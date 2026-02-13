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

    # Common, explicit usage requirements.
    #
    # - itlabai_features: project-wide feature defines
    # - itlabai_openmp: optional OpenMP linkage (empty if disabled)
    get_target_property(_tgt_type ${target_name} TYPE)
    if(_tgt_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries(${target_name} INTERFACE itlabai_features itlabai_openmp)
    else()
        target_link_libraries(${target_name} PUBLIC itlabai_features itlabai_openmp)
    endif()
endfunction()
