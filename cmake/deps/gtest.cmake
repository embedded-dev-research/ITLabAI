set(GTEST_PREFIX "${ITLABAI_EXTERNAL_ROOT}/gtest")
set(GTEST_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/gtest")
set(GTEST_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/gtest")

if(ITLABAI_USE_SYSTEM_DEPS)
    find_package(GTest QUIET CONFIG)
endif()

if(NOT GTest_FOUND)
    find_package(Threads REQUIRED)
    ExternalProject_Add(gtest_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/googletest"
        BINARY_DIR "${GTEST_BUILD_DIR}"
        INSTALL_DIR "${GTEST_INSTALL_DIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_GMOCK=ON
            -DINSTALL_GTEST=ON
            -DBUILD_SHARED_LIBS=OFF
            $<$<BOOL:${MSVC}>:-Dgtest_force_shared_crt=ON>
            $<$<BOOL:${MSVC}>:-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL>
        BUILD_BYPRODUCTS
            ${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    add_dependencies(itlabai_external gtest_external)

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
else()
    if(TARGET GTest::gtest AND NOT TARGET gtest)
        add_library(gtest INTERFACE IMPORTED)
        target_link_libraries(gtest INTERFACE GTest::gtest)
    endif()
    if(TARGET GTest::gtest_main AND NOT TARGET gtest_main)
        add_library(gtest_main INTERFACE IMPORTED)
        target_link_libraries(gtest_main INTERFACE GTest::gtest_main)
    endif()
endif()
