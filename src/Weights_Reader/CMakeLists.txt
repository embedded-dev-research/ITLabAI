file(GLOB READER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(reader_lib STATIC)
target_sources(reader_lib PRIVATE ${READER_SOURCES} ${READER_HEADERS})
target_include_directories(reader_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(reader_lib PUBLIC nlohmann_json::nlohmann_json)
itlabai_target_defaults(reader_lib)
itlabai_link_externals(reader_lib)
