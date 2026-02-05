add_library(perf_lib INTERFACE)
target_sources(perf_lib INTERFACE
    $<BUILD_INTERFACE:${PERF_HEADERS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/perf/benchmarking.hpp>
)
target_include_directories(perf_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
itlabai_link_externals(perf_lib openmp)

add_library(perf_anchor OBJECT benchmarking.cpp)
itlabai_target_defaults(perf_anchor)
set_target_properties(perf_anchor PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(perf_anchor PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(perf_anchor PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_dependencies(perf_lib perf_anchor)
