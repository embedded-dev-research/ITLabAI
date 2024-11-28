file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")

execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_apps=OFF
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)

add_executable(ACCImgNet accimgnet.cpp)

find_package( OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ACCImgNet ${OpenCV_LIBS} )
target_link_libraries( ACCImgNet TBB::tbb)
target_link_libraries( ACCImgNet layers_lib)
target_link_libraries( ACCImgNet gtest_main)

if (WIN32)
add_custom_command(TARGET ACCImgNet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/app/ACCImgNet/${CMAKE_BUILD_TYPE}/")
endif()
