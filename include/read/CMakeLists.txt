set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/Reader_img.hpp")
set(SRC_FILES "${CMAKE_SOURCE_DIR}/../itlab_2023/src/read/Reader_img.cpp")
add_library(MyInclude STATIC ${INCLUDE_HEADERS} ${SRC_FILES})
set_target_properties(MyInclude PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(MyInclude PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(OpenCV_DIR C:\\sema\\opencv\\build )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( MyInclude ${OpenCV_LIBS} )






