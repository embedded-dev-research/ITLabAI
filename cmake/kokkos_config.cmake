include(ExternalProject)

set(KOKKOS_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/kokkos_build")
set(KOKKOS_INSTALL_DIR "${CMAKE_BINARY_DIR}/3rdparty/kokkos_install")

ExternalProject_Add(
    kokkos_external
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/kokkos"
    BINARY_DIR "${KOKKOS_BUILD_DIR}"
    INSTALL_DIR "${KOKKOS_INSTALL_DIR}"
    
    CMAKE_ARGS
        -G "${CMAKE_GENERATOR}"
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR}
        
        -DKokkos_ENABLE_SERIAL=ON
        -DKokkos_ARCH_NATIVE=OFF
        -DKokkos_ENABLE_OPENMP=OFF
        -DKokkos_ENABLE_THREADS=ON
        -DKokkos_ENABLE_CUDA=OFF
        -DKokkos_ENABLE_HIP=OFF
        -DKokkos_ENABLE_TESTS=OFF
        -DKokkos_ENABLE_EXAMPLES=OFF
        
        -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON
        -DKokkos_ENABLE_LIBDL=OFF
    
    BUILD_COMMAND ${CMAKE_COMMAND} --build "${KOKKOS_BUILD_DIR}" --config ${CMAKE_BUILD_TYPE} -j${NPROC}
    
    INSTALL_COMMAND ${CMAKE_COMMAND} --install "${KOKKOS_BUILD_DIR}" --config ${CMAKE_BUILD_TYPE}
    
    BUILD_ALWAYS OFF
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)

set(Kokkos_DIR "${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos" CACHE PATH "Path to Kokkos CMake config")
