file(GLOB_RECURSE ITLABAI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/inference/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/model_read/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_read/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_layer/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_layer_parall_version/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarking/*.cpp"
)

add_executable(itlabai_run_test_test ${ITLABAI_TEST_SOURCES})
add_executable(ITLabAI::RunTestTest ALIAS itlabai_run_test_test)
set_target_properties(itlabai_run_test_test PROPERTIES OUTPUT_NAME "itlabai_run_test")

target_link_libraries(itlabai_run_test_test PUBLIC
    itlabai_perf_lib
    itlabai_layers_lib
    itlabai_layers_onednn_lib
    itlabai_reader_image_lib
    itlabai_reader_lib
    itlabai_graph_lib
    itlabai_graph_transformations_lib
    Kokkos_imported
    TBB_unified
    OpenCV::opencv_world
    gtest_main
    gtest
)
itlabai_link_externals(itlabai_run_test_test openmp)
itlabai_link_externals(itlabai_run_test_test gtest opencv tbb kokkos openmp)
add_dependencies(itlabai_run_test_test opencv_external)

target_include_directories(itlabai_run_test_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(itlabai_run_test_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")
itlabai_target_defaults(itlabai_run_test_test)
itlabai_apply_runtime_rpath(itlabai_run_test_test)

if (WIN32)
    target_compile_definitions(itlabai_run_test_test PRIVATE _CRT_SECURE_NO_WARNINGS)
    add_custom_command(TARGET itlabai_run_test_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:dnnl>
            $<TARGET_FILE_DIR:itlabai_run_test_test>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:OpenCV::opencv_world>
            $<TARGET_FILE_DIR:itlabai_run_test_test>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TBB::tbb>
            $<TARGET_FILE_DIR:itlabai_run_test_test>
    )
endif()

add_test(UnitTests $<TARGET_FILE:itlabai_run_test_test>)

target_compile_definitions(itlabai_run_test_test PRIVATE TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/test/model_read/json_for_test")
target_compile_definitions(itlabai_run_test_test PRIVATE IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")
target_compile_definitions(itlabai_run_test_test PRIVATE IMAGENET_LABELS="${ITLABAI_TEST_DATA_DIR}/imagenet-labels.txt")
target_compile_definitions(itlabai_run_test_test PRIVATE TESTS_BINARY_PATH="${ITLABAI_TEST_DATA_DIR}")
add_dependencies(itlabai_run_test_test opencv_external)
