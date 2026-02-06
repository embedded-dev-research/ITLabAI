file(GLOB ACC_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB ACC_HPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
list(FILTER ACC_CPP EXCLUDE REGEX "accuracy_check\\.cpp$")

add_library(ACCLib STATIC ${ACC_HPP} ${ACC_CPP})

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

file(GLOB ACC_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/accuracy_check.cpp")
add_executable(Accuracy_Check ${ACC_MAIN})
target_link_libraries(Accuracy_Check PRIVATE ACCLib OpenCV::opencv_world)
itlabai_link_externals(Accuracy_Check openmp)
itlabai_target_defaults(Accuracy_Check)
itlabai_apply_runtime_rpath(Accuracy_Check)
