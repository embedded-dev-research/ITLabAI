file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(OpenMP REQUIRED)

add_executable(run_test ${TEST_SRC_FILES})
if(OpenMP_CXX_FOUND)
	target_link_libraries(run_test PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)
target_link_libraries(run_test PUBLIC gtest_main)
add_test(UnitTests ${CMAKE_CURRENT_BINARY_DIR}/run_test)
