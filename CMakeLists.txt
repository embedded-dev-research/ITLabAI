cmake_minimum_required(VERSION 3.20)

project(ITLabAI VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_SOURCE_DIR}/cmake/ProjectDefaults.cmake)

option(ITLABAI_BUILD_APPS "Build applications" ON)
option(ITLABAI_BUILD_TESTS "Build tests" ON)
option(ITLABAI_ENABLE_OPENMP "Enable OpenMP support" ON)
option(ITLABAI_ENABLE_KOKKOS "Build with Kokkos external" ON)
option(ITLABAI_USE_SYSTEM_DEPS "Use system packages when available" OFF)
option(ITLABAI_FETCH_TEST_DATA "Download test data during build" OFF)
set(ITLABAI_TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/docs/input" CACHE PATH "Directory containing sample/test data")
option(BUILD_TESTING "Enable building tests" ${ITLABAI_BUILD_TESTS})
option(ITLABAI_WARN_AS_ERROR "Treat warnings as errors for project targets" ON)
option(ITLABAI_AUTOINCLUDE_EXTERNALS "Automatically add include directories for external deps when linked via helper" ON)

option(ENABLE_STATISTIC_TENSORS "Enable statistic tensors" OFF)
option(ENABLE_STATISTIC_TIME "Enable statistic time" OFF)
option(ENABLE_STATISTIC_WEIGHTS "Enable statistic weights" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ITLABAI_FEATURE_DEFS "")
if(ENABLE_STATISTIC_TENSORS)
    list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_TENSORS)
endif()
if(ENABLE_STATISTIC_TIME)
    list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_TIME)
endif()
if(ENABLE_STATISTIC_WEIGHTS)
    list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_WEIGHTS)
endif()

if(ITLABAI_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found - enabling parallel support")
        list(APPEND ITLABAI_FEATURE_DEFS HAS_OPENMP)
        if(OpenMP_CXX_FLAGS)
            add_compile_options(${OpenMP_CXX_FLAGS})
        endif()
        if(OpenMP_EXE_LINKER_FLAGS)
            add_link_options(${OpenMP_EXE_LINKER_FLAGS})
        endif()
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# External dependencies (isolated builds)
include(cmake/deps.cmake)

if(ITLABAI_FETCH_TEST_DATA)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(fetch_test_data
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ITLABAI_TEST_DATA_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/fetch_test_data.py --dest ${ITLABAI_TEST_DATA_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Downloading sample/test data"
    )
endif()

if(NOT WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    foreach(CONFIG "" _DEBUG _RELEASE)
        set("CMAKE_ARCHIVE_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/lib")
        set("CMAKE_LIBRARY_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/lib")
        set("CMAKE_RUNTIME_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/bin")
    endforeach()
endif()

# RPATH for bundled externals
set(_itlabai_rpath "")
itlabai_set_rpath(
    "${ONEDNN_INSTALL_DIR}/lib"
    "${OPENCV_INSTALL_DIR}/lib"
    "${TBB_INSTALL_DIR}/lib"
    "${KOKKOS_INSTALL_DIR}/lib"
)

add_subdirectory(include)
add_subdirectory(src)
if(ITLABAI_BUILD_APPS)
    add_subdirectory(app)
endif()
if(BUILD_TESTING AND ITLABAI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Aggregate interface target
if(NOT TARGET itlabai)
    add_library(itlabai INTERFACE)
    add_library(ITLabAI::itlabai ALIAS itlabai)
endif()
target_link_libraries(itlabai INTERFACE graph_lib graphT_lib layers_lib layers_oneDNN_lib perf_lib reader_lib)
target_include_directories(itlabai INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(ITLABAI_FEATURE_DEFS)
    target_compile_definitions(itlabai INTERFACE ${ITLABAI_FEATURE_DEFS})
endif()

# Install/export
install(TARGETS graph_lib graphT_lib layers_lib layers_oneDNN_lib perf_lib reader_lib itlabai
    EXPORT ITLabAITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/ITLabAIConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI
)

install(EXPORT ITLabAITargets
    NAMESPACE ITLabAI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI
)
