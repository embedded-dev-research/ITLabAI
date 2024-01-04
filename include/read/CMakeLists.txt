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
