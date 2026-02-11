cmake_minimum_required(VERSION 3.20)

project(ITLabAI VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProjectDefaults.cmake)

option(ITLABAI_BUILD_APPS "Build applications" ON)
option(ITLABAI_ENABLE_OPENMP "Enable OpenMP support" ON)
option(ITLABAI_ENABLE_KOKKOS "Build with Kokkos external" ON)
option(ITLABAI_FETCH_TEST_DATA "Download test data during configure" ON)
set(_itlabai_default_test_data_dir "${CMAKE_BINARY_DIR}/input")
if(NOT DEFINED ITLABAI_TEST_DATA_DIR)
    set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data")
elseif(ITLABAI_TEST_DATA_DIR STREQUAL "${PROJECT_SOURCE_DIR}/docs/input")
    # Auto-migrate old default to build-local data dir.
    set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data" FORCE)
endif()
option(BUILD_TESTING "Enable building tests" ON)
set(ITLABAI_DOCS_DIR "${PROJECT_SOURCE_DIR}/docs" CACHE PATH "Project docs root")
set(ITLABAI_MODELS_JSON_DIR "${ITLABAI_DOCS_DIR}/jsons" CACHE PATH "Directory with generated model json files")
set(ITLABAI_IMAGENET_LABELS_FILE "${ITLABAI_DOCS_DIR}/imagenet1000_clsidx_to_labels.json" CACHE FILEPATH "ImageNet labels file")
set(ITLABAI_MNIST_DIR "${ITLABAI_DOCS_DIR}/mnist/mnist/test" CACHE PATH "MNIST dataset directory")
set(ITLABAI_IMAGENET_ACC_DIR "${ITLABAI_DOCS_DIR}/ImageNet/test" CACHE PATH "ImageNet accuracy dataset directory")
set(ITLABAI_TEST_MODEL_JSON_DIR "${PROJECT_SOURCE_DIR}/test/model_read/json_for_test" CACHE PATH "Test model json directory")

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
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# External dependencies
include(cmake/deps.cmake)

if(ITLABAI_FETCH_TEST_DATA)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ITLABAI_TEST_DATA_DIR}
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/fetch_test_data.py --dest ${ITLABAI_TEST_DATA_DIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE _itlabai_fetch_data_result
    )
    if(NOT _itlabai_fetch_data_result EQUAL 0)
        message(FATAL_ERROR "Failed to fetch test data. Disable with -DITLABAI_FETCH_TEST_DATA=OFF if needed.")
    endif()
endif()

add_subdirectory(src)
if(ITLABAI_BUILD_APPS)
    add_subdirectory(app)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# Aggregate interface target
if(NOT TARGET itlabai)
    add_library(itlabai INTERFACE)
    add_library(ITLabAI::itlabai ALIAS itlabai)
endif()
target_include_directories(itlabai INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(ITLABAI_FEATURE_DEFS)
    target_compile_definitions(itlabai INTERFACE ${ITLABAI_FEATURE_DEFS})
endif()

# Install/export
target_link_libraries(itlabai INTERFACE
    itlabai_graph_lib
    itlabai_graph_transformations_lib
    itlabai_layers_lib
    itlabai_layers_onednn_lib
    itlabai_perf_lib
    itlabai_reader_lib
)

install(TARGETS
    itlabai_graph_lib
    itlabai_graph_transformations_lib
    itlabai_layers_lib
    itlabai_layers_onednn_lib
    itlabai_perf_lib
    itlabai_reader_lib
    itlabai
    EXPORT ITLabAITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

# Bundle vendored third-party runtime and headers into package install.
install(DIRECTORY "${ONEDNN_INSTALL_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty/onednn")
install(DIRECTORY "${TBB_INSTALL_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty/tbb")
install(DIRECTORY "${OPENCV_INSTALL_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty/opencv")
if(ITLABAI_ENABLE_KOKKOS)
    install(DIRECTORY "${KOKKOS_INSTALL_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty/kokkos")
endif()
install(DIRECTORY "${PROJECT_SOURCE_DIR}/3rdparty/Json/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty/json")

set(_itlabai_vendor_lib_dir "${CMAKE_INSTALL_LIBDIR}/itlabai/thirdparty/lib")
install(FILES "$<TARGET_FILE:dnnl>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
install(FILES "$<TARGET_FILE:TBB::tbb>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
install(FILES "$<TARGET_FILE:OpenCV::opencv_world>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
if(ITLABAI_ENABLE_KOKKOS)
    install(FILES "$<TARGET_FILE:kokkoscore_external>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
    install(FILES "$<TARGET_FILE:kokkoscontainers_external>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
endif()

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
