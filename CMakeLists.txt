cmake_minimum_required(VERSION 3.20)

project(ITLabAI VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProjectDefaults.cmake)

# Options
option(ITLABAI_BUILD_APPS "Build applications" ON)
option(ITLABAI_ENABLE_OPENMP "Enable OpenMP support" ON)
option(ITLABAI_ENABLE_KOKKOS "Build with Kokkos external" ON)
option(ITLABAI_FETCH_TEST_DATA "Fetch test data during build when needed" ON)
option(ITLABAI_WERROR "Treat warnings as errors for ITLabAI targets" ON)

option(BUILD_TESTING "Enable building tests" ON)

# Single-config generators (Ninja/Makefiles) need an explicit build type.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Language standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project data / assets (used by apps/tests).
set(ITLABAI_DOCS_DIR "${PROJECT_SOURCE_DIR}/docs" CACHE PATH "Project docs root")
set(ITLABAI_MODELS_JSON_DIR "${ITLABAI_DOCS_DIR}/jsons" CACHE PATH "Directory with generated model json files")
set(ITLABAI_IMAGENET_LABELS_FILE "${ITLABAI_DOCS_DIR}/imagenet1000_clsidx_to_labels.json" CACHE FILEPATH "ImageNet labels file")
set(ITLABAI_MNIST_DIR "${ITLABAI_DOCS_DIR}/mnist/mnist/test" CACHE PATH "MNIST dataset directory")
set(ITLABAI_IMAGENET_ACC_DIR "${ITLABAI_DOCS_DIR}/ImageNet/test" CACHE PATH "ImageNet accuracy dataset directory")
set(ITLABAI_TEST_MODEL_JSON_DIR "${PROJECT_SOURCE_DIR}/test/model_read/json_for_test" CACHE PATH "Test model json directory")

# Test data location. Default is build-local to keep working tree clean.
set(_itlabai_default_test_data_dir "${CMAKE_BINARY_DIR}/input")
if(NOT DEFINED ITLABAI_TEST_DATA_DIR)
  set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data")
elseif(ITLABAI_TEST_DATA_DIR STREQUAL "${PROJECT_SOURCE_DIR}/docs/input")
  # Auto-migrate old default to build-local data dir.
  set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data" FORCE)
endif()

# Deprecated options (kept for cache compatibility). Use ITLABAI_* variants.
option(ENABLE_STATISTIC_TENSORS "Deprecated: use ITLABAI_ENABLE_STATISTIC_TENSORS" OFF)
option(ENABLE_STATISTIC_TIME "Deprecated: use ITLABAI_ENABLE_STATISTIC_TIME" OFF)
option(ENABLE_STATISTIC_WEIGHTS "Deprecated: use ITLABAI_ENABLE_STATISTIC_WEIGHTS" OFF)
mark_as_advanced(ENABLE_STATISTIC_TENSORS ENABLE_STATISTIC_TIME ENABLE_STATISTIC_WEIGHTS)

option(ITLABAI_ENABLE_STATISTIC_TENSORS "Enable statistic tensors" ${ENABLE_STATISTIC_TENSORS})
option(ITLABAI_ENABLE_STATISTIC_TIME "Enable statistic time" ${ENABLE_STATISTIC_TIME})
option(ITLABAI_ENABLE_STATISTIC_WEIGHTS "Enable statistic weights" ${ENABLE_STATISTIC_WEIGHTS})

# Feature defines
set(ITLABAI_FEATURE_DEFS "")
if(ITLABAI_ENABLE_STATISTIC_TENSORS)
  list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_TENSORS)
endif()
if(ITLABAI_ENABLE_STATISTIC_TIME)
  list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_TIME)
endif()
if(ITLABAI_ENABLE_STATISTIC_WEIGHTS)
  list(APPEND ITLABAI_FEATURE_DEFS ENABLE_STATISTIC_WEIGHTS)
endif()

if(ITLABAI_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  list(APPEND ITLABAI_FEATURE_DEFS HAS_OPENMP)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Third-party dependencies (isolated via ExternalProject).
include(cmake/deps.cmake)

# Test data (used by tests and demo apps). Always define the target to avoid conditional
# add_dependencies() blocks in subdirectories.
if(ITLABAI_FETCH_TEST_DATA)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_target(itlabai_fetch_test_data
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ITLABAI_TEST_DATA_DIR}"
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/fetch_test_data.py" --dest "${ITLABAI_TEST_DATA_DIR}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Fetching ITLabAI test data into ${ITLABAI_TEST_DATA_DIR}"
    VERBATIM
  )
else()
  add_custom_target(itlabai_fetch_test_data)
endif()

# Project targets
add_subdirectory(src)
if(ITLABAI_BUILD_APPS)
  add_subdirectory(app)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Packaging.cmake")

