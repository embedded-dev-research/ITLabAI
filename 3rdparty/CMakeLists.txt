add_subdirectory(googletest)

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build")

add_custom_target(build_tbb ALL
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/TBB" -B "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build"
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DTBB_TEST=OFF
        -DCMAKE_CXX_FLAGS="-w"
        -DCMAKE_C_FLAGS="-w"
        -DTBB_OUTPUT_DIR_BASE=tbb
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build"
)

add_custom_target(build_opencv ALL
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_opencv_apps=OFF
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
