set(ONEDNN_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/onednn")
set(ONEDNN_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/onednn")

set(_onednn_build_type "${ITLABAI_EXTERNAL_BUILD_TYPE}")

set(_onednn_byproducts "")
if(MSVC)
    list(APPEND _onednn_byproducts
        ${ONEDNN_INSTALL_DIR}/lib/dnnl.lib
        ${ONEDNN_INSTALL_DIR}/bin/dnnl.dll
    )
else()
    list(APPEND _onednn_byproducts
        ${ONEDNN_INSTALL_DIR}/lib/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
endif()

itlabai_external_add(
    NAME onednn_external
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/oneDNN"
    BINARY_DIR "${ONEDNN_BUILD_DIR}"
    INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
    DEPENDS tbb_external
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${_onednn_build_type}
        -DDNNL_BUILD_TESTS=OFF
        -DDNNL_BUILD_EXAMPLES=OFF
        -DDNNL_BUILD_DOC=OFF
        -DDNNL_BUILD_GRAPH=OFF
        -DDNNL_ENABLE_WORKLOAD=INFERENCE
        -DDNNL_ENABLE_PRIMITIVE=ALL
        -DDNNL_CPU_RUNTIME=TBB
        -DTBB_ROOT=${TBB_INSTALL_DIR}
        -DDNNL_LIBRARY_TYPE=SHARED
        -DBUILD_SHARED_LIBS=ON
        ${ITLABAI_EXTERNAL_TOOLCHAIN_ARGS}
        ${ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX}
    BUILD_BYPRODUCTS ${_onednn_byproducts}
)

if(MSVC)
    set(_dnnl_lib "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib")
    set(_dnnl_dll "${ONEDNN_INSTALL_DIR}/bin/dnnl.dll")
else()
    set(_dnnl_lib "${ONEDNN_INSTALL_DIR}/lib/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

file(MAKE_DIRECTORY "${ONEDNN_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${ONEDNN_INSTALL_DIR}/lib")

add_library(dnnl SHARED IMPORTED GLOBAL)
if(MSVC)
    set_target_properties(dnnl PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_dnnl_dll}"
        IMPORTED_LOCATION_DEBUG "${_dnnl_dll}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_dnnl_dll}"
        IMPORTED_LOCATION_MINSIZEREL "${_dnnl_dll}"
        IMPORTED_IMPLIB "${_dnnl_lib}"
        IMPORTED_IMPLIB_RELEASE "${_dnnl_lib}"
        IMPORTED_IMPLIB_DEBUG "${_dnnl_lib}"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_dnnl_lib}"
        IMPORTED_IMPLIB_MINSIZEREL "${_dnnl_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INSTALL_DIR}/include"
    )
else()
    set_target_properties(dnnl PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_dnnl_lib}"
        IMPORTED_LOCATION_DEBUG "${_dnnl_lib}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_dnnl_lib}"
        IMPORTED_LOCATION_MINSIZEREL "${_dnnl_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INSTALL_DIR}/include"
    )
endif()
target_link_libraries(dnnl INTERFACE TBB::tbb)
add_dependencies(dnnl onednn_external)
