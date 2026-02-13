include_guard(GLOBAL)

# Derive compile definitions from options and detected features.
# The resulting list is consumed by the itlabai_options interface target.

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
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found - enabling parallel support")
    list(APPEND ITLABAI_FEATURE_DEFS HAS_OPENMP)
  endif()
endif()

