include_guard(GLOBAL)

# Shared compile options/defines for all in-tree ITLabAI targets.

if(NOT TARGET itlabai_options)
  add_library(itlabai_options INTERFACE)
endif()

if(ITLABAI_FEATURE_DEFS)
  target_compile_definitions(itlabai_options INTERFACE ${ITLABAI_FEATURE_DEFS})
endif()

target_compile_features(itlabai_options INTERFACE cxx_std_20)

