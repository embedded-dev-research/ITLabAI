file(GLOB_RECURSE reader_src *.cpp)
add_library(reader_lib STATIC "${READER_HEADERS}" "${reader_src}")

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Json/include)
target_include_directories(reader_lib PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/Json/include)
