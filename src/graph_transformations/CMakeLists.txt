file(GLOB GRAPHT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(graphT_lib STATIC)
target_sources(graphT_lib PRIVATE ${GRAPHT_SOURCES} ${GRAPHT_HEADERS})
target_include_directories(graphT_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
itlabai_link_externals(graphT_lib tbb kokkos openmp)
itlabai_target_defaults(graphT_lib)
