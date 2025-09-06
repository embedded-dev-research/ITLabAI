set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/reader_img_s.cpp")
add_library(ReadLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})


set_target_properties(ReadLib PROPERTIES LINKER_LANGUAGE CXX)

find_package( OpenCV REQUIRED PATHS "${OPENCV_BUILD_DIR}" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ReadLib ${OpenCV_LIBS} )
target_link_libraries( ReadLib TBB_unified)

add_executable(Reader reader_img.cpp)
target_link_libraries(Reader ReadLib)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DIMAGE1_PATH="${CMAKE_CURRENT_BINARY_DIR}/image.jpg")
