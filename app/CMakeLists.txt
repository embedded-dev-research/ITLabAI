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
add_custom_command(TARGET Reader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/3rdparty/TBB/build/msvc_19.36_cxx_64_md_debug/tbb12_debug.dll"
            "$<TARGET_FILE_DIR:Reader>")
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
