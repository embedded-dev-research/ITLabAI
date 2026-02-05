set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/reader_img_s.cpp")
add_library(ReadLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(ReadLib PROPERTIES LINKER_LANGUAGE CXX)

itlabai_link_externals(ReadLib opencv tbb openmp)
target_link_libraries(ReadLib PUBLIC OpenCV::opencv_world TBB_unified)
itlabai_target_defaults(ReadLib)
add_dependencies(ReadLib opencv_external)
target_compile_definitions(ReadLib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

add_executable(Reader reader_img.cpp)
target_link_libraries(Reader PRIVATE ReadLib OpenCV::opencv_world)
itlabai_target_defaults(Reader)
