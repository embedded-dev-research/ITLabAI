# Common helper functions and defaults for project targets

function(itlabai_target_defaults target_name)
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
            string(TOUPPER "${_cfg}" _cfg_upper)
            string(TOLOWER "${_cfg}" _cfg_lower)
            set_target_properties(${target_name} PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/lib/${_cfg_lower}"
                LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/lib/${_cfg_lower}"
                RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin/${_cfg_lower}"
            )
        endforeach()
    else()
        set_target_properties(${target_name} PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()

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

function(itlabai_use_opencv target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} OpenCV::opencv_world)
endfunction()

function(itlabai_use_tbb target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} TBB::tbb)
endfunction()

function(itlabai_use_onednn target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} dnnl)
endfunction()

function(itlabai_use_kokkos target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} Kokkos_imported)
    if(MSVC)
        # Suppress Kokkos header warning C4702 only on targets that use Kokkos.
        target_compile_options(${target_name} ${_scope} /wd4702)
    endif()
endfunction()

function(itlabai_use_openmp target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} OpenMP::OpenMP_CXX)
endfunction()

function(itlabai_use_gtest target_name)
    itlabai_use_externals_scope(_scope ${target_name})
    target_link_libraries(${target_name} ${_scope} gtest_main gtest)
endfunction()

function(itlabai_use_externals_scope out_var target_name)
    get_target_property(_tgt_type ${target_name} TYPE)
    if(_tgt_type STREQUAL "INTERFACE_LIBRARY")
        set(${out_var} INTERFACE PARENT_SCOPE)
    else()
        set(${out_var} PUBLIC PARENT_SCOPE)
    endif()
endfunction()

function(itlabai_link_externals target_name)
    foreach(ext IN LISTS ARGN)
        if(ext STREQUAL "opencv")
            itlabai_use_opencv(${target_name})
        elseif(ext STREQUAL "tbb")
            itlabai_use_tbb(${target_name})
        elseif(ext STREQUAL "onednn")
            itlabai_use_onednn(${target_name})
        elseif(ext STREQUAL "kokkos")
            itlabai_use_kokkos(${target_name})
        elseif(ext STREQUAL "openmp")
            if(ITLABAI_ENABLE_OPENMP)
                itlabai_use_openmp(${target_name})
            endif()
        elseif(ext STREQUAL "gtest")
            itlabai_use_gtest(${target_name})
        endif()
    endforeach()
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
