# Создаем папку сборки для OpenCV
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")
# Вызываем команды для сборки OpenCV через execute_process
execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build"
)
set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.hpp")
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/read/reader_img.cpp")
add_library(ReadLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(ReadLib PROPERTIES LINKER_LANGUAGE CXX)

find_package( OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ReadLib ${OpenCV_LIBS} )

#Скачиваем изображение
file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    #EXPECTED_HASH SHA256=expected_hash_value
    STATUS status_code
    LOG log_file
)

set(OpenCV_DLL_PATH "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin")  # Замените путем к фактической папке с .dll
set_property(DIRECTORY APPEND PROPERTY ENVIRONMENT "PATH=${OpenCV_DLL_PATH}")

