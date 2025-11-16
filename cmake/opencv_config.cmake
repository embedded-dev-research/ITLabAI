set(OPENCV_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/opencv_build")
file(MAKE_DIRECTORY "${OPENCV_BUILD_DIR}")

set(_opencv_bin_dir "${OPENCV_BUILD_DIR}/bin")
file(MAKE_DIRECTORY "${_opencv_bin_dir}")
if (WIN32)
    foreach(_cfg Debug Release RelWithDebInfo MinSizeRel)
        file(MAKE_DIRECTORY "${_opencv_bin_dir}/${_cfg}")
    endforeach()
endif()

unset(_opencv_bin_dir)

execute_process(
    COMMAND ${CMAKE_COMMAND} 
        -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" 
        -B "${OPENCV_BUILD_DIR}" 
        -G "${CMAKE_GENERATOR}"
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} 
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} 
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} 
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} 
        -DBUILD_PERF_TESTS=OFF 
        -DBUILD_TESTS=OFF 
        -DBUILD_opencv_apps=OFF
    WORKING_DIRECTORY "${OPENCV_BUILD_DIR}"
)

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${OPENCV_BUILD_DIR}" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${OPENCV_BUILD_DIR}"
)
