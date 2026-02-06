file(GLOB READER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(itlabai_reader_lib STATIC)
add_library(ITLabAI::ReaderLib ALIAS itlabai_reader_lib)
target_sources(itlabai_reader_lib PRIVATE ${READER_SOURCES} ${READER_HEADERS})
target_include_directories(itlabai_reader_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai_reader_lib PUBLIC nlohmann_json::nlohmann_json)
itlabai_target_defaults(itlabai_reader_lib)
itlabai_link_externals(itlabai_reader_lib)
