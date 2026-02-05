set(ONEDNN_PREFIX "${ITLABAI_EXTERNAL_ROOT}/onednn")
set(ONEDNN_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/onednn")
set(ONEDNN_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/onednn")

if(ITLABAI_USE_SYSTEM_DEPS)
    find_package(dnnl QUIET CONFIG)
endif()

if(NOT dnnl_FOUND)
    set(_onednn_depends "")
    if(TARGET tbb_external)
        set(_onednn_depends tbb_external)
    endif()

    ExternalProject_Add(onednn_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/oneDNN"
        BINARY_DIR "${ONEDNN_BUILD_DIR}"
        INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
        DEPENDS ${_onednn_depends}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
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
        BUILD_BYPRODUCTS
            ${ONEDNN_INSTALL_DIR}/lib/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    add_dependencies(itlabai_external onednn_external)

    if(MSVC)
        set(_dnnl_lib "${ONEDNN_INSTALL_DIR}/lib/dnnl.dll")
    else()
        set(_dnnl_lib "${ONEDNN_INSTALL_DIR}/lib/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    file(MAKE_DIRECTORY "${ONEDNN_INSTALL_DIR}/include")
    file(MAKE_DIRECTORY "${ONEDNN_INSTALL_DIR}/lib")

    add_library(dnnl STATIC IMPORTED GLOBAL)
    set_target_properties(dnnl PROPERTIES
        IMPORTED_LOCATION_RELEASE "${_dnnl_lib}"
        IMPORTED_LOCATION_DEBUG "${_dnnl_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INSTALL_DIR}/include"
    )
    set_target_properties(dnnl PROPERTIES
        IMPORTED_LOCATION_RELWITHDEBINFO "${_dnnl_lib}"
        IMPORTED_LOCATION_MINSIZEREL "${_dnnl_lib}"
    )
    target_link_libraries(dnnl INTERFACE TBB::tbb)
    add_dependencies(dnnl onednn_external)
else()
    if(TARGET dnnl::dnnl)
        add_library(dnnl INTERFACE IMPORTED)
        target_link_libraries(dnnl INTERFACE dnnl::dnnl)
    endif()
endif()
