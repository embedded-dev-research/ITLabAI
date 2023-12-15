set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/Reader_img.hpp")

file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp")

add_library(MyInclude STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

target_include_directories(MyInclude PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})



