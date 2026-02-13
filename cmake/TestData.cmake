include_guard(GLOBAL)

if(NOT ITLABAI_FETCH_TEST_DATA)
  return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(itlabai_fetch_test_data
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ITLABAI_TEST_DATA_DIR}"
  COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/fetch_test_data.py" --dest "${ITLABAI_TEST_DATA_DIR}"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Fetching ITLabAI test data into ${ITLABAI_TEST_DATA_DIR}"
  VERBATIM
)

