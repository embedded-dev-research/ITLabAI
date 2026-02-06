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

add_executable(run_test ${ITLABAI_TEST_SOURCES})

target_link_libraries(run_test PUBLIC
    perf_lib
    layers_lib
    layers_oneDNN_lib
    ReadLib
    reader_lib
    graph_lib
    graphT_lib
    Kokkos_imported
    TBB_unified
    OpenCV::opencv_world
    gtest_main
    gtest
)
itlabai_link_externals(run_test openmp)
itlabai_link_externals(run_test gtest opencv tbb kokkos openmp)
add_dependencies(run_test opencv_external)

target_include_directories(run_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")
itlabai_target_defaults(run_test)
itlabai_apply_runtime_rpath(run_test)

if (WIN32)
    target_compile_definitions(run_test PRIVATE _CRT_SECURE_NO_WARNINGS)
    add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:dnnl>
            $<TARGET_FILE_DIR:run_test>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:OpenCV::opencv_world>
            $<TARGET_FILE_DIR:run_test>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TBB::tbb>
            $<TARGET_FILE_DIR:run_test>
    )
endif()

add_test(UnitTests ${CMAKE_BINARY_DIR}/bin/run_test)

target_compile_definitions(run_test PRIVATE TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/test/model_read/json_for_test")
target_compile_definitions(run_test PRIVATE IMAGE1_PATH="${ITLABAI_TEST_DATA_DIR}/lena.jpg")
target_compile_definitions(run_test PRIVATE IMAGENET_LABELS="${ITLABAI_TEST_DATA_DIR}/imagenet-labels.txt")
target_compile_definitions(run_test PRIVATE TESTS_BINARY_PATH="${ITLABAI_TEST_DATA_DIR}")
add_dependencies(run_test opencv_external)
