file(GLOB READER_IMG_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
file(GLOB READER_IMG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*_s.cpp")
add_library(itlabai_reader_image_lib STATIC ${READER_IMG_HEADERS} ${READER_IMG_SOURCES})
add_library(ITLabAI::ReaderImageLib ALIAS itlabai_reader_image_lib)

set_target_properties(itlabai_reader_image_lib PROPERTIES LINKER_LANGUAGE CXX)

if(TARGET itlabai_fetch_test_data)
    add_dependencies(itlabai_reader_image_lib itlabai_fetch_test_data)
endif()

itlabai_link_externals(itlabai_reader_image_lib opencv tbb openmp)
itlabai_target_defaults(itlabai_reader_image_lib)
target_compile_definitions(itlabai_reader_image_lib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

file(GLOB READER_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/reader_img.cpp")
add_executable(itlabai_reader_exe ${READER_MAIN})
add_executable(ITLabAI::ReaderExe ALIAS itlabai_reader_exe)
set_target_properties(itlabai_reader_exe PROPERTIES OUTPUT_NAME "itlabai_reader")
target_link_libraries(itlabai_reader_exe PRIVATE itlabai_reader_image_lib)
itlabai_target_defaults(itlabai_reader_exe)
itlabai_apply_runtime_rpath(itlabai_reader_exe)
