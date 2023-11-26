cmake_minimum_required(VERSION 3.20)

project(itlab_2023)

enable_testing()

include_directories("include")

add_subdirectory(3rdparty)
add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

file(GLOB_RECURSE ALL_SOURCES_FILES src/*.cpp include/*.h include/*.hpp)
add_library(app_libs STATIC ${ALL_SOURCES_FILES})