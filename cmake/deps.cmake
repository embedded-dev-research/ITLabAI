include(ExternalProject)

# Root directories for all external builds/installs
set(ITLABAI_EXTERNAL_ROOT "${CMAKE_BINARY_DIR}/_external" CACHE PATH "Root for external project builds")

# ExternalProject build dirs are not cleaned by `cmake --fresh`, and switching compilers
# in the same build dir can leave stale external caches behind. Isolate external build/install
# trees by toolchain to avoid mixing artifacts and accidentally installing to system paths.
function(itlabai_sanitize_identifier out_var in_str)
    set(_s "${in_str}")
    if(_s STREQUAL "")
        set(_s "unknown")
    endif()
    string(REGEX REPLACE "[^A-Za-z0-9_.-]+" "_" _s "${_s}")
    set(${out_var} "${_s}" PARENT_SCOPE)
endfunction()

set(_itlabai_toolchain_id "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
itlabai_sanitize_identifier(ITLABAI_EXTERNAL_TOOLCHAIN_ID "${_itlabai_toolchain_id}")

# Do NOT cache these: they must track the active toolchain reliably.
set(ITLABAI_EXTERNAL_BUILD_ROOT "${ITLABAI_EXTERNAL_ROOT}/build/${ITLABAI_EXTERNAL_TOOLCHAIN_ID}")
set(ITLABAI_EXTERNAL_INSTALL_ROOT "${ITLABAI_EXTERNAL_ROOT}/install/${ITLABAI_EXTERNAL_TOOLCHAIN_ID}")

function(itlabai_external_default_build_type out_var)
    set(_bt "${CMAKE_BUILD_TYPE}")
    if(NOT _bt)
        set(_bt "Release")
    endif()
    set(${out_var} "${_bt}" PARENT_SCOPE)
endfunction()

function(itlabai_external_add)
    set(options)
    set(one_value_args NAME SOURCE_DIR BINARY_DIR INSTALL_DIR)
    set(multi_value_args DEPENDS CMAKE_ARGS CMAKE_CACHE_ARGS BUILD_BYPRODUCTS)
    cmake_parse_arguments(EP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT EP_NAME)
        message(FATAL_ERROR "itlabai_external_add: NAME is required")
    endif()
    if(NOT EP_SOURCE_DIR)
        message(FATAL_ERROR "itlabai_external_add: SOURCE_DIR is required")
    endif()
    if(NOT EP_BINARY_DIR)
        message(FATAL_ERROR "itlabai_external_add: BINARY_DIR is required")
    endif()
    if(NOT EP_INSTALL_DIR)
        message(FATAL_ERROR "itlabai_external_add: INSTALL_DIR is required")
    endif()

    # Force a non-system install prefix for every external. This prevents any external
    # from using its default (often /usr/local) even when reconfiguring is skipped.
    set(_itlabai_ep_cache_args
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_INSTALL_DIR}
    )

    ExternalProject_Add(${EP_NAME}
        SOURCE_DIR "${EP_SOURCE_DIR}"
        BINARY_DIR "${EP_BINARY_DIR}"
        INSTALL_DIR "${EP_INSTALL_DIR}"
        DEPENDS ${EP_DEPENDS}
        CMAKE_ARGS ${EP_CMAKE_ARGS}
        CMAKE_CACHE_ARGS ${_itlabai_ep_cache_args} ${EP_CMAKE_CACHE_ARGS}
        BUILD_BYPRODUCTS ${EP_BUILD_BYPRODUCTS}
    )
    add_dependencies(itlabai_external ${EP_NAME})
endfunction()

function(itlabai_external_collect_toolchain_args out_var)
    set(_args "")
    set(_itlabai_cc "")
    set(_itlabai_cxx "")
    if(DEFINED CMAKE_C_COMPILER AND NOT CMAKE_C_COMPILER STREQUAL "")
        set(_itlabai_cc "${CMAKE_C_COMPILER}")
    elseif(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
        set(_itlabai_cc "$ENV{CC}")
    endif()
    if(DEFINED CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL "")
        set(_itlabai_cxx "${CMAKE_CXX_COMPILER}")
    elseif(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
        set(_itlabai_cxx "$ENV{CXX}")
    endif()
    if(_itlabai_cc)
        list(APPEND _args -DCMAKE_C_COMPILER=${_itlabai_cc})
    endif()
    if(_itlabai_cxx)
        list(APPEND _args -DCMAKE_CXX_COMPILER=${_itlabai_cxx})
    endif()
    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

itlabai_external_collect_toolchain_args(ITLABAI_EXTERNAL_TOOLCHAIN_ARGS)

set(ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX "")
set(ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY "")
list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF)
list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF)
if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX -DCMAKE_C_FLAGS=/WX- -DCMAKE_CXX_FLAGS=/WX-)
    list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY -DCMAKE_CXX_FLAGS=/WX-)
else()
    list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX -DCMAKE_C_FLAGS=-Wno-error -DCMAKE_CXX_FLAGS=-Wno-error)
    list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY -DCMAKE_CXX_FLAGS=-Wno-error)
endif()

add_custom_target(itlabai_external) # aggregator for externals

include(${CMAKE_CURRENT_LIST_DIR}/deps/tbb.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/deps/onednn.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/deps/opencv.cmake)
if(ITLABAI_ENABLE_KOKKOS)
    include(${CMAKE_CURRENT_LIST_DIR}/deps/kokkos.cmake)
else()
    if(NOT TARGET Kokkos_imported)
        add_library(Kokkos_imported INTERFACE IMPORTED GLOBAL)
    endif()
endif()
include(${CMAKE_CURRENT_LIST_DIR}/deps/gtest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/deps/json.cmake)

# Compatibility aliases for existing target names
if(NOT TARGET TBB_unified)
    add_library(TBB_unified INTERFACE IMPORTED GLOBAL)
    target_link_libraries(TBB_unified INTERFACE TBB::tbb)
endif()
