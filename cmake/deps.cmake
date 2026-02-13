include(ExternalProject)

# Third-party dependencies are built via ExternalProject into an isolated tree under:
#   ${CMAKE_BINARY_DIR}/_external/{build,install}/<toolchain-id>/
#
# This avoids accidentally installing to system locations (e.g. /usr/local) and avoids
# stale caches when switching compilers in the same build directory.

set(ITLABAI_EXTERNAL_ROOT "${CMAKE_BINARY_DIR}/_external" CACHE PATH "Root for external project builds")

# Toolchain-scoped external roots (keeps ExternalProject caches isolated per compiler).
set(ITLABAI_EXTERNAL_TOOLCHAIN_ID "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
string(MAKE_C_IDENTIFIER "${ITLABAI_EXTERNAL_TOOLCHAIN_ID}" ITLABAI_EXTERNAL_TOOLCHAIN_ID)

set(ITLABAI_EXTERNAL_BUILD_ROOT "${ITLABAI_EXTERNAL_ROOT}/build/${ITLABAI_EXTERNAL_TOOLCHAIN_ID}")
set(ITLABAI_EXTERNAL_INSTALL_ROOT "${ITLABAI_EXTERNAL_ROOT}/install/${ITLABAI_EXTERNAL_TOOLCHAIN_ID}")

# External projects are configured as single-config builds; pick a build type once and reuse it.
set(ITLABAI_EXTERNAL_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NOT ITLABAI_EXTERNAL_BUILD_TYPE)
  set(ITLABAI_EXTERNAL_BUILD_TYPE "Release")
endif()

# Propagate compiler selection into ExternalProject builds.
set(ITLABAI_EXTERNAL_TOOLCHAIN_ARGS "")
if(DEFINED CMAKE_C_COMPILER AND NOT CMAKE_C_COMPILER STREQUAL "")
  list(APPEND ITLABAI_EXTERNAL_TOOLCHAIN_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif()
if(DEFINED CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL "")
  list(APPEND ITLABAI_EXTERNAL_TOOLCHAIN_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif()

# Keep external warnings from killing the build; ITLabAI itself can still be built with Werror.
set(ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
)
set(ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
)
if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX -DCMAKE_C_FLAGS=/WX- -DCMAKE_CXX_FLAGS=/WX-)
  list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY -DCMAKE_CXX_FLAGS=/WX-)
else()
  list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX -DCMAKE_C_FLAGS=-Wno-error -DCMAKE_CXX_FLAGS=-Wno-error)
  list(APPEND ITLABAI_EXTERNAL_WARNING_ARGS_CXX_ONLY -DCMAKE_CXX_FLAGS=-Wno-error)
endif()

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

  # Force a non-system install prefix for every external.
  set(ep_cache_args
    -DCMAKE_INSTALL_PREFIX:PATH=${EP_INSTALL_DIR}
  )

  ExternalProject_Add(${EP_NAME}
    SOURCE_DIR "${EP_SOURCE_DIR}"
    BINARY_DIR "${EP_BINARY_DIR}"
    INSTALL_DIR "${EP_INSTALL_DIR}"
    DEPENDS ${EP_DEPENDS}
    CMAKE_ARGS ${EP_CMAKE_ARGS}
    CMAKE_CACHE_ARGS ${ep_cache_args} ${EP_CMAKE_CACHE_ARGS}
    BUILD_BYPRODUCTS ${EP_BUILD_BYPRODUCTS}
  )
endfunction()

include(${CMAKE_CURRENT_LIST_DIR}/deps/tbb.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/deps/onednn.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/deps/opencv.cmake)

if(ITLABAI_ENABLE_KOKKOS)
  include(${CMAKE_CURRENT_LIST_DIR}/deps/kokkos.cmake)
else()
  add_library(Kokkos_imported INTERFACE IMPORTED GLOBAL)
endif()

if(BUILD_TESTING)
  include(${CMAKE_CURRENT_LIST_DIR}/deps/gtest.cmake)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/deps/json.cmake)

