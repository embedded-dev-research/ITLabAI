set(KOKKOS_PREFIX "${ITLABAI_EXTERNAL_ROOT}/kokkos")
set(KOKKOS_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/kokkos")
set(KOKKOS_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/kokkos")

itlabai_external_default_build_type(_kokkos_build_type)
set(_kokkos_openmp_flag OFF)
set(_kokkos_threads_flag ON)
if(ITLABAI_ENABLE_OPENMP AND OpenMP_FOUND AND NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
    set(_kokkos_openmp_flag ON)
    set(_kokkos_threads_flag OFF)
endif()
if(MSVC)
    # MSVC OpenMP is limited to 2.0; disable OpenMP for Kokkos.
    set(_kokkos_openmp_flag OFF)
    set(_kokkos_threads_flag ON)
endif()

itlabai_external_add(
    NAME kokkos_external
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/kokkos"
    BINARY_DIR "${KOKKOS_BUILD_DIR}"
    INSTALL_DIR "${KOKKOS_INSTALL_DIR}"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=${_kokkos_build_type}
        -DKokkos_ENABLE_SERIAL=ON
        -DKokkos_ENABLE_THREADS=${_kokkos_threads_flag}
        -DKokkos_ENABLE_OPENMP=${_kokkos_openmp_flag}
        -DKokkos_ENABLE_CUDA=OFF
        -DKokkos_ENABLE_HIP=OFF
        -DKokkos_ENABLE_TESTS=OFF
        -DKokkos_ENABLE_EXAMPLES=OFF
        -DBUILD_SHARED_LIBS=OFF
        ${ITLABAI_EXTERNAL_TOOLCHAIN_ARGS}
        ${ITLABAI_EXTERNAL_WARNING_ARGS}
    BUILD_BYPRODUCTS
        ${KOKKOS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kokkoscore${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${KOKKOS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kokkoscontainers${CMAKE_STATIC_LIBRARY_SUFFIX}
)

if(MSVC)
    set(_kokkos_core "${KOKKOS_INSTALL_DIR}/lib/kokkoscore.lib")
    set(_kokkos_cont "${KOKKOS_INSTALL_DIR}/lib/kokkoscontainers.lib")
else()
    set(_kokkos_core "${KOKKOS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kokkoscore${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_kokkos_cont "${KOKKOS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kokkoscontainers${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

file(MAKE_DIRECTORY "${KOKKOS_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${KOKKOS_INSTALL_DIR}/lib")

add_library(kokkoscore_external STATIC IMPORTED GLOBAL)
set_target_properties(kokkoscore_external PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_kokkos_core}"
    IMPORTED_LOCATION_DEBUG "${_kokkos_core}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_kokkos_core}"
    IMPORTED_LOCATION_MINSIZEREL "${_kokkos_core}"
    INTERFACE_INCLUDE_DIRECTORIES "${KOKKOS_INSTALL_DIR}/include"
)
add_dependencies(kokkoscore_external kokkos_external)

add_library(kokkoscontainers_external STATIC IMPORTED GLOBAL)
set_target_properties(kokkoscontainers_external PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_kokkos_cont}"
    IMPORTED_LOCATION_DEBUG "${_kokkos_cont}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_kokkos_cont}"
    IMPORTED_LOCATION_MINSIZEREL "${_kokkos_cont}"
    INTERFACE_INCLUDE_DIRECTORIES "${KOKKOS_INSTALL_DIR}/include"
)
add_dependencies(kokkoscontainers_external kokkos_external)
if(NOT TARGET Kokkos_imported)
    add_library(Kokkos_imported INTERFACE IMPORTED GLOBAL)
endif()
add_dependencies(Kokkos_imported kokkos_external)
target_link_libraries(Kokkos_imported INTERFACE kokkoscore_external kokkoscontainers_external)
