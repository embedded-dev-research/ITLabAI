cmake_minimum_required(VERSION 3.7) 

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")

execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" --config "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)

add_executable(accuracy_app "./accuracy_app/accuracy_check.cpp")

find_package( OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( accuracy_app ${OpenCV_LIBS} )

# ------------------------------------------

add_executable(Reader reader_img.cpp)
add_executable(alexnet alexnet.cpp)
target_link_libraries(alexnet AlexNetLib)
target_link_libraries(Reader ReadLib)

if (WIN32)
add_custom_command(TARGET Reader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/app/${CMAKE_BUILD_TYPE}/")
add_custom_command(TARGET alexnet POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
            "$<TARGET_FILE_DIR:alexnet>")
endif()
file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DIMAGE1_PATH="${CMAKE_CURRENT_BINARY_DIR}/image.jpg")
file(DOWNLOAD
    "https://tinyurl.com/2w6eh53v"
    "${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DMODEL1_PATH="${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite")
