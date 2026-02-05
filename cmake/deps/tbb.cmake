set(TBB_PREFIX "${ITLABAI_EXTERNAL_ROOT}/tbb")
set(TBB_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/tbb")
set(TBB_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/tbb")

if(ITLABAI_USE_SYSTEM_DEPS)
    find_package(TBB QUIET CONFIG)
endif()

if(NOT TBB_FOUND)
    ExternalProject_Add(tbb_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/TBB"
        BINARY_DIR "${TBB_BUILD_DIR}"
        INSTALL_DIR "${TBB_INSTALL_DIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${TBB_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DTBB_TEST=OFF
            -DTBB_EXAMPLES=OFF
            -DTBB_STRICT=OFF
        BUILD_BYPRODUCTS
            ${TBB_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tbb${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    add_dependencies(itlabai_external tbb_external)

    if(MSVC)
        set(_tbb_lib "${TBB_INSTALL_DIR}/lib/tbb_static.lib")
    else()
        set(_tbb_lib "${TBB_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tbb${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()

    file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/include")
    file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/lib")

    add_library(TBB::tbb STATIC IMPORTED GLOBAL)
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_tbb_lib}"
        IMPORTED_LOCATION_DEBUG "${_tbb_lib}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_tbb_lib}"
        IMPORTED_LOCATION_MINSIZEREL "${_tbb_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
    )
    add_dependencies(TBB::tbb tbb_external)
else()
    # Use system-provided TBB target
endif()
