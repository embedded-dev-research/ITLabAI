add_executable(Reader reader_img.cpp)

target_link_libraries(Reader ReadLib)

add_custom_command(TARGET Reader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/Debug/"
        "${CMAKE_BINARY_DIR}/app/Debug/")

add_custom_command(TARGET Reader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/Debug/"
        "${CMAKE_BINARY_DIR}/app/Release/")

#Скачиваем изображение
file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    #EXPECTED_HASH SHA256=expected_hash_value
    STATUS status_code
    LOG log_file
)
