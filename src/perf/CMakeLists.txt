add_library(itlabai_perf_lib INTERFACE)
add_library(ITLabAI::PerfLib ALIAS itlabai_perf_lib)
target_sources(itlabai_perf_lib INTERFACE
    $<BUILD_INTERFACE:${PERF_HEADERS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/perf/benchmarking.hpp>
)
target_include_directories(itlabai_perf_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
itlabai_link_externals(itlabai_perf_lib openmp)

add_library(itlabai_perf_anchor_obj OBJECT benchmarking.cpp)
itlabai_target_defaults(itlabai_perf_anchor_obj)
set_target_properties(itlabai_perf_anchor_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ITLABAI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(itlabai_perf_anchor_obj PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(itlabai_perf_anchor_obj PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_dependencies(itlabai_perf_lib itlabai_perf_anchor_obj)
