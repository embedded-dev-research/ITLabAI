set(KOKKOS_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/kokkos_build")
file(MAKE_DIRECTORY "${KOKKOS_BUILD_DIR}")

execute_process(
    COMMAND ${CMAKE_COMMAND} 
        -S "${CMAKE_SOURCE_DIR}/3rdparty/kokkos" 
        -B "${KOKKOS_BUILD_DIR}"
        -G "${CMAKE_GENERATOR}"
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${KOKKOS_BUILD_DIR}/install
        
        -DKokkos_ENABLE_SERIAL=ON
        -DKokkos_ARCH_NATIVE=OFF
        -DKokkos_ENABLE_OPENMP=OFF
        -DKokkos_ENABLE_CUDA=OFF
        -DKokkos_ENABLE_HIP=OFF
        -DKokkos_ENABLE_TESTS=OFF
        -DKokkos_ENABLE_EXAMPLES=OFF
        
        -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON
        -DKokkos_ENABLE_LIBDL=OFF
    WORKING_DIRECTORY "${KOKKOS_BUILD_DIR}"
    RESULT_VARIABLE config_result
)

if(NOT config_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure Kokkos")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${KOKKOS_BUILD_DIR}" --config ${CMAKE_BUILD_TYPE} -j${NPROC}
    RESULT_VARIABLE build_result
    WORKING_DIRECTORY "${KOKKOS_BUILD_DIR}"
)

if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build Kokkos")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${KOKKOS_BUILD_DIR}" --config ${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY "${KOKKOS_BUILD_DIR}"
)
