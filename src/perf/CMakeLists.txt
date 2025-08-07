file(GLOB_RECURSE perf_src *.cpp)
add_library(perf_lib STATIC "${PERF_HEADERS}" "${perf_src}")

find_package(OpenMP REQUIRED)
target_link_libraries(perf_lib PUBLIC OpenMP::OpenMP_CXX)
