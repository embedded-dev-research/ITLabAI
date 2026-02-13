file(GLOB READER_IMG_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
file(GLOB READER_IMG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*_s.cpp")
add_library(itlabai_reader_image_lib STATIC ${READER_IMG_HEADERS} ${READER_IMG_SOURCES})

set_target_properties(itlabai_reader_image_lib PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(itlabai_reader_image_lib PUBLIC OpenCV::opencv_world TBB::tbb)
itlabai_target_defaults(itlabai_reader_image_lib)
target_compile_definitions(itlabai_reader_image_lib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

file(GLOB READER_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.cpp")
add_executable(itlabai_reader_exe ${READER_MAIN})
set_target_properties(itlabai_reader_exe PROPERTIES OUTPUT_NAME "itlabai_reader")
target_link_libraries(itlabai_reader_exe PRIVATE itlabai_reader_image_lib)
add_dependencies(itlabai_reader_exe itlabai_fetch_test_data)
itlabai_target_defaults(itlabai_reader_exe)
