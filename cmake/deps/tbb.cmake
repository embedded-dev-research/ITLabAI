set(TBB_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/tbb")
set(TBB_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/tbb")

string(TOLOWER "${ITLABAI_EXTERNAL_BUILD_TYPE}" tbb_build_type_lower)
set(tbb_cmake_args ${ITLABAI_EXTERNAL_TOOLCHAIN_ARGS} ${ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX})

if(WIN32)
  set(tbb_debug_suffix "")
  if(tbb_build_type_lower STREQUAL "debug")
    set(tbb_debug_suffix "_debug")
  endif()

  set(tbb_release_lib "${TBB_INSTALL_DIR}/lib/tbb12.lib")
  set(tbb_release_dll "${TBB_INSTALL_DIR}/bin/tbb12.dll")
  set(tbb_debug_lib "${TBB_INSTALL_DIR}/lib/tbb12_debug.lib")
  set(tbb_debug_dll "${TBB_INSTALL_DIR}/bin/tbb12_debug.dll")

  set(tbb_lib "${TBB_INSTALL_DIR}/lib/tbb12${tbb_debug_suffix}.lib")
  set(tbb_dll "${TBB_INSTALL_DIR}/bin/tbb12${tbb_debug_suffix}.dll")

  set(tbb_byproducts
    "${tbb_lib}"
    "${tbb_dll}"
  )
else()
  set(tbb_debug_suffix "")
  if(tbb_build_type_lower STREQUAL "debug")
    set(tbb_debug_suffix "_debug")
  endif()

  set(tbb_release_lib "${TBB_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tbb${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(tbb_debug_lib "${TBB_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tbb_debug${CMAKE_SHARED_LIBRARY_SUFFIX}")

  set(tbb_byproducts
    "${TBB_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tbb${tbb_debug_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
endif()

itlabai_external_add(
    NAME tbb_external
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/TBB"
    BINARY_DIR "${TBB_BUILD_DIR}"
    INSTALL_DIR "${TBB_INSTALL_DIR}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${ITLABAI_EXTERNAL_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=ON
        -DTBB_TEST=OFF
        -DTBB_EXAMPLES=OFF
        -DTBB_STRICT=OFF
        # IPO/LTO with clang can produce LLVM bitcode objects that are later linked
        # by a non-clang driver in some configurations, causing "file format not recognized".
        -DTBB_ENABLE_IPO=OFF
        ${tbb_cmake_args}
    BUILD_BYPRODUCTS ${tbb_byproducts}
)

file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${TBB_INSTALL_DIR}/lib")

add_library(TBB::tbb SHARED IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION "${tbb_dll}"
        IMPORTED_LOCATION_RELEASE "${tbb_release_dll}"
        IMPORTED_LOCATION_DEBUG "${tbb_debug_dll}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${tbb_release_dll}"
        IMPORTED_LOCATION_MINSIZEREL "${tbb_release_dll}"
        IMPORTED_IMPLIB "${tbb_lib}"
        IMPORTED_IMPLIB_RELEASE "${tbb_release_lib}"
        IMPORTED_IMPLIB_DEBUG "${tbb_debug_lib}"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${tbb_release_lib}"
        IMPORTED_IMPLIB_MINSIZEREL "${tbb_release_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
    )
else()
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION "${tbb_release_lib}"
        IMPORTED_LOCATION_RELEASE "${tbb_release_lib}"
        IMPORTED_LOCATION_DEBUG "${tbb_debug_lib}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${tbb_release_lib}"
        IMPORTED_LOCATION_MINSIZEREL "${tbb_release_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
    )
endif()
add_dependencies(TBB::tbb tbb_external)
