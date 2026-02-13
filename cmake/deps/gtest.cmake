set(GTEST_PREFIX "${ITLABAI_EXTERNAL_ROOT}/gtest")
set(GTEST_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/gtest")
set(GTEST_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/gtest")

find_package(Threads REQUIRED)
set(_gtest_build_type "${ITLABAI_EXTERNAL_BUILD_TYPE}")
set(_gtest_cmake_args "")
if(MSVC)
    set(_gtest_msvc_runtime "MultiThreadedDLL")
    if(_gtest_build_type STREQUAL "Debug")
        set(_gtest_msvc_runtime "MultiThreadedDebugDLL")
    endif()
    list(APPEND _gtest_cmake_args
        -Dgtest_force_shared_crt=ON
        -DCMAKE_MSVC_RUNTIME_LIBRARY=${_gtest_msvc_runtime}
    )
endif()
itlabai_external_add(
    NAME gtest_external
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/googletest"
    BINARY_DIR "${GTEST_BUILD_DIR}"
    INSTALL_DIR "${GTEST_INSTALL_DIR}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${_gtest_build_type}
        -DBUILD_GMOCK=ON
        -DINSTALL_GTEST=ON
        -DBUILD_SHARED_LIBS=OFF
        ${_gtest_cmake_args}
        ${ITLABAI_EXTERNAL_TOOLCHAIN_ARGS}
        ${ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX}
    BUILD_BYPRODUCTS
        ${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
)

if(MSVC)
    set(_gtest_lib "${GTEST_INSTALL_DIR}/lib/gtest.lib")
    set(_gtest_main_lib "${GTEST_INSTALL_DIR}/lib/gtest_main.lib")
else()
    set(_gtest_lib "${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_gtest_main_lib "${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

file(MAKE_DIRECTORY "${GTEST_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${GTEST_INSTALL_DIR}/lib")

add_library(gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_gtest_lib}"
    IMPORTED_LOCATION_DEBUG "${_gtest_lib}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_gtest_lib}"
    IMPORTED_LOCATION_MINSIZEREL "${_gtest_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INSTALL_DIR}/include"
)
target_link_libraries(gtest INTERFACE Threads::Threads)
add_dependencies(gtest gtest_external)

add_library(gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_gtest_main_lib}"
    IMPORTED_LOCATION_DEBUG "${_gtest_main_lib}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_gtest_main_lib}"
    IMPORTED_LOCATION_MINSIZEREL "${_gtest_main_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INSTALL_DIR}/include"
)
target_link_libraries(gtest_main INTERFACE gtest Threads::Threads)
add_dependencies(gtest_main gtest_external)
