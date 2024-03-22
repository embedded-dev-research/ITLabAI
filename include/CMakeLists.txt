file(GLOB_RECURSE graph_headers graph/*.h graph/*.hpp)
set(GRAPH_HEADERS "${graph_headers}" PARENT_SCOPE)

file(GLOB_RECURSE layers_headers layers/*.h layers/*.hpp)
set(LAYERS_HEADERS "${layers_headers}" PARENT_SCOPE)

file(GLOB_RECURSE perf_headers perf/*.h perf/*.hpp)
set(PERF_HEADERS "${perf_headers}" PARENT_SCOPE)

add_subdirectory(read)

find_package( OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ReadLib ${OpenCV_LIBS} )