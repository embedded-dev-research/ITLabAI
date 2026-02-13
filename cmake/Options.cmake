include_guard(GLOBAL)

# User-facing build options and cache variables.

option(ITLABAI_BUILD_APPS "Build applications" ON)
option(ITLABAI_ENABLE_OPENMP "Enable OpenMP support" ON)
option(ITLABAI_ENABLE_KOKKOS "Build with Kokkos external" ON)
option(ITLABAI_FETCH_TEST_DATA "Fetch test data (build target) during build when needed" ON)
option(ITLABAI_WERROR "Treat warnings as errors for ITLabAI targets" ON)

option(BUILD_TESTING "Enable building tests" ON)

# Test data location. Default is build-local to keep working tree clean.
set(_itlabai_default_test_data_dir "${CMAKE_BINARY_DIR}/input")
if(NOT DEFINED ITLABAI_TEST_DATA_DIR)
  set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data")
elseif(ITLABAI_TEST_DATA_DIR STREQUAL "${PROJECT_SOURCE_DIR}/docs/input")
  # Auto-migrate old default to build-local data dir.
  set(ITLABAI_TEST_DATA_DIR "${_itlabai_default_test_data_dir}" CACHE PATH "Directory containing sample/test data" FORCE)
endif()

# Project data / assets (used by apps/tests).
set(ITLABAI_DOCS_DIR "${PROJECT_SOURCE_DIR}/docs" CACHE PATH "Project docs root")
set(ITLABAI_MODELS_JSON_DIR "${ITLABAI_DOCS_DIR}/jsons" CACHE PATH "Directory with generated model json files")
set(ITLABAI_IMAGENET_LABELS_FILE "${ITLABAI_DOCS_DIR}/imagenet1000_clsidx_to_labels.json" CACHE FILEPATH "ImageNet labels file")
set(ITLABAI_MNIST_DIR "${ITLABAI_DOCS_DIR}/mnist/mnist/test" CACHE PATH "MNIST dataset directory")
set(ITLABAI_IMAGENET_ACC_DIR "${ITLABAI_DOCS_DIR}/ImageNet/test" CACHE PATH "ImageNet accuracy dataset directory")
set(ITLABAI_TEST_MODEL_JSON_DIR "${PROJECT_SOURCE_DIR}/test/model_read/json_for_test" CACHE PATH "Test model json directory")

# Deprecated options (kept for cache compatibility). Use ITLABAI_* variants.
option(ENABLE_STATISTIC_TENSORS "Deprecated: use ITLABAI_ENABLE_STATISTIC_TENSORS" OFF)
option(ENABLE_STATISTIC_TIME "Deprecated: use ITLABAI_ENABLE_STATISTIC_TIME" OFF)
option(ENABLE_STATISTIC_WEIGHTS "Deprecated: use ITLABAI_ENABLE_STATISTIC_WEIGHTS" OFF)
mark_as_advanced(ENABLE_STATISTIC_TENSORS ENABLE_STATISTIC_TIME ENABLE_STATISTIC_WEIGHTS)

option(ITLABAI_ENABLE_STATISTIC_TENSORS "Enable statistic tensors" ${ENABLE_STATISTIC_TENSORS})
option(ITLABAI_ENABLE_STATISTIC_TIME "Enable statistic time" ${ENABLE_STATISTIC_TIME})
option(ITLABAI_ENABLE_STATISTIC_WEIGHTS "Enable statistic weights" ${ENABLE_STATISTIC_WEIGHTS})

