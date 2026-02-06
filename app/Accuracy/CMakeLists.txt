file(GLOB ACC_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB ACC_HPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
list(FILTER ACC_CPP EXCLUDE REGEX "accuracy_check\\.cpp$")

add_library(itlabai_acc_lib STATIC ${ACC_HPP} ${ACC_CPP})
add_library(ITLabAI::AccLib ALIAS itlabai_acc_lib)

set_target_properties(itlabai_acc_lib PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(itlabai_acc_lib
    PUBLIC
        itlabai_layers_lib
        OpenCV::opencv_world
        TBB_unified
        Kokkos_imported
)
itlabai_link_externals(itlabai_acc_lib opencv tbb gtest kokkos openmp)
itlabai_target_defaults(itlabai_acc_lib)
add_dependencies(itlabai_acc_lib opencv_external)
target_compile_definitions(itlabai_acc_lib PUBLIC IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")

file(GLOB ACC_MAIN CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/accuracy_check.cpp")
add_executable(itlabai_accuracy_check_exe ${ACC_MAIN})
add_executable(ITLabAI::AccuracyCheckExe ALIAS itlabai_accuracy_check_exe)
target_link_libraries(itlabai_accuracy_check_exe PRIVATE itlabai_acc_lib OpenCV::opencv_world)
itlabai_link_externals(itlabai_accuracy_check_exe openmp)
itlabai_target_defaults(itlabai_accuracy_check_exe)
itlabai_apply_runtime_rpath(itlabai_accuracy_check_exe)
