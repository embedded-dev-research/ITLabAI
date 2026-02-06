set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/acc.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/acc.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/acc_anchor.cpp")
add_library(ACCLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(ACCLib PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(ACCLib
    PUBLIC
        layers_lib
        OpenCV::opencv_world
        TBB_unified
        Kokkos_imported
)
itlabai_link_externals(ACCLib opencv tbb gtest kokkos openmp)
itlabai_target_defaults(ACCLib)
add_dependencies(ACCLib opencv_external)
target_compile_definitions(ACCLib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

add_executable(Accuracy_Check accuracy_check.cpp)
target_link_libraries(Accuracy_Check PRIVATE ACCLib OpenCV::opencv_world)
itlabai_link_externals(Accuracy_Check openmp)
itlabai_target_defaults(Accuracy_Check)
itlabai_apply_runtime_rpath(Accuracy_Check)
