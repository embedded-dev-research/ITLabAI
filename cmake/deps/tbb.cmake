set(TBB_PREFIX "${ITLABAI_EXTERNAL_ROOT}/tbb")
set(TBB_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/tbb")
set(TBB_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/tbb")

if(ITLABAI_USE_SYSTEM_DEPS)
    find_package(TBB QUIET CONFIG)
endif()

if(NOT TBB_FOUND)
    set(_tbb_cmake_args "")
    if(MSVC)
        list(APPEND _tbb_cmake_args
            -DCMAKE_C_COMPILER=cl
            -DCMAKE_CXX_COMPILER=cl
        )
    elseif(WIN32)
        get_filename_component(_clang_dir "${CMAKE_C_COMPILER}" DIRECTORY)
        set(_clang_cl "${_clang_dir}/clang-cl.exe")
        if(EXISTS "${_clang_cl}")
            list(APPEND _tbb_cmake_args
                -DCMAKE_C_COMPILER=${_clang_cl}
                -DCMAKE_CXX_COMPILER=${_clang_cl}
            )
        endif()
    endif()

    if(WIN32)
        set(_tbb_lib_name "tbb12")
        set(_tbb_lib "${TBB_INSTALL_DIR}/lib/${_tbb_lib_name}.lib")
        set(_tbb_dll "${TBB_INSTALL_DIR}/bin/${_tbb_lib_name}.dll")
        set(_tbb_byproducts
            "${_tbb_lib}"
            "${_tbb_dll}"
        )
    else()
        set(_tbb_byproducts
            "${TBB_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tbb${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
        set(_tbb_lib "${TBB_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tbb${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    ExternalProject_Add(tbb_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/TBB"
        BINARY_DIR "${TBB_BUILD_DIR}"
        INSTALL_DIR "${TBB_INSTALL_DIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${TBB_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=ON
            -DTBB_TEST=OFF
            -DTBB_EXAMPLES=OFF
            -DTBB_STRICT=OFF
            ${_tbb_cmake_args}
        BUILD_BYPRODUCTS
            ${_tbb_byproducts}
    )
    add_dependencies(itlabai_external tbb_external)

    file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/include")
    file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/lib")

    add_library(TBB::tbb SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_target_properties(TBB::tbb PROPERTIES
            IMPORTED_LOCATION_RELEASE "${_tbb_dll}"
            IMPORTED_LOCATION_DEBUG "${_tbb_dll}"
            IMPORTED_LOCATION_RELWITHDEBINFO "${_tbb_dll}"
            IMPORTED_LOCATION_MINSIZEREL "${_tbb_dll}"
            IMPORTED_IMPLIB "${_tbb_lib}"
            IMPORTED_IMPLIB_RELEASE "${_tbb_lib}"
            IMPORTED_IMPLIB_DEBUG "${_tbb_lib}"
            IMPORTED_IMPLIB_RELWITHDEBINFO "${_tbb_lib}"
            IMPORTED_IMPLIB_MINSIZEREL "${_tbb_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
        )
    else()
        set_target_properties(TBB::tbb PROPERTIES
            IMPORTED_LOCATION_RELEASE "${_tbb_lib}"
            IMPORTED_LOCATION_DEBUG "${_tbb_lib}"
            IMPORTED_LOCATION_RELWITHDEBINFO "${_tbb_lib}"
            IMPORTED_LOCATION_MINSIZEREL "${_tbb_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
        )
    endif()
    add_dependencies(TBB::tbb tbb_external)
else()
    # Use system-provided TBB target
endif()
