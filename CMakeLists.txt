cmake_minimum_required(VERSION 3.20)

project(ITLabAI VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProjectDefaults.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Options.cmake)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Features.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TargetsCommon.cmake)

# External dependencies
include(cmake/deps.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestData.cmake)

add_subdirectory(src)
if(ITLABAI_BUILD_APPS)
    add_subdirectory(app)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/InTreeAliases.cmake)

# Aggregate interface target
if(NOT TARGET itlabai)
    add_library(itlabai INTERFACE)
    add_library(ITLabAI::itlabai ALIAS itlabai)
endif()
target_include_directories(itlabai INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(itlabai INTERFACE itlabai_options)

# Install/export
target_link_libraries(itlabai INTERFACE
    itlabai_graph_lib
    itlabai_graph_transformations_lib
    itlabai_layers_lib
    itlabai_layers_onednn_lib
    itlabai_perf_lib
    itlabai_reader_lib
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Packaging.cmake")
