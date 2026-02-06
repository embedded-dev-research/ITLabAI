file(GLOB READER_IMG_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
file(GLOB READER_IMG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*_s.cpp")
add_library(ReadLib STATIC ${READER_IMG_HEADERS} ${READER_IMG_SOURCES})

set_target_properties(ReadLib PROPERTIES LINKER_LANGUAGE CXX)

itlabai_link_externals(ReadLib opencv tbb openmp)
target_link_libraries(ReadLib PUBLIC OpenCV::opencv_world TBB_unified)
itlabai_target_defaults(ReadLib)
add_dependencies(ReadLib opencv_external)
target_compile_definitions(ReadLib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

file(GLOB READER_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.cpp")
add_executable(Reader ${READER_MAIN})
target_link_libraries(Reader PRIVATE ReadLib OpenCV::opencv_world)
itlabai_target_defaults(Reader)
itlabai_apply_runtime_rpath(Reader)
