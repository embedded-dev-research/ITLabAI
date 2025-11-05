set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/acc.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/acc.cpp")
add_library(ACCLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(ReadLib PROPERTIES LINKER_LANGUAGE CXX)

find_package( OpenCV REQUIRED PATHS "${OPENCV_BUILD_DIR}" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ACCLib ${OpenCV_LIBS} )
target_link_libraries( ACCLib TBB_unified)
target_link_libraries( ACCLib layers_lib)
target_link_libraries( ACCLib gtest_main)

add_executable(Accuracy_Check accuracy_check.cpp)
target_link_libraries(Accuracy_Check ACCLib)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DIMAGE1_PATH="${CMAKE_CURRENT_BINARY_DIR}/image.jpg")
