include(ExternalProject)

# Root directories for all external builds/installs
set(ITLABAI_EXTERNAL_ROOT "${CMAKE_BINARY_DIR}/_external" CACHE PATH "Root for external project builds")
set(ITLABAI_EXTERNAL_BUILD_ROOT "${ITLABAI_EXTERNAL_ROOT}/build" CACHE PATH "External build trees")
set(ITLABAI_EXTERNAL_INSTALL_ROOT "${ITLABAI_EXTERNAL_ROOT}/install" CACHE PATH "External install trees")

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
