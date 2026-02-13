set(OPENCV_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/opencv_min")
set(OPENCV_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/opencv_min")

set(OPENCV_COMPONENTS core imgproc imgcodecs highgui world)
set(OPENCV_COMPONENTS_ESC "core\\;imgproc\\;imgcodecs\\;highgui\\;world")

set(_opencv_build_type "${ITLABAI_EXTERNAL_BUILD_TYPE}")
set(OPENCV_BUILD_OPTS
    -DBUILD_TESTS=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_DOCS=OFF
    -DBUILD_opencv_apps=OFF
    -DBUILD_opencv_dnn=OFF
    -DBUILD_opencv_python=OFF
    -DBUILD_JAVA=OFF
    -DBUILD_opencv_world=ON
    -DBUILD_IPP_IW=OFF
    -DINSTALL_C_EXAMPLES=OFF
    -DINSTALL_PYTHON_EXAMPLES=OFF
    -DINSTALL_CASCADES=OFF
    -DINSTALL_TESTS=OFF
    -DOPENCV_GENERATE_PKGCONFIG=OFF
    -DOPENCV_INSTALL_FFMPEG=OFF
    -DOPENCV_DISABLE_ADE=ON
    -DOPENCV_ENABLE_NONFREE=OFF
    -DOPENCV_DOWNLOAD_PATH=${PROJECT_SOURCE_DIR}/3rdparty/opencv_downloads
    -DCMAKE_CXX_STANDARD=17
)
set(OPENCV_FEATURE_OPTS
    -DWITH_TBB=OFF
    -DWITH_IPP=OFF
    -DWITH_OPENEXR=OFF
    -DWITH_AVIF=OFF
    -DWITH_WEBP=OFF
    -DWITH_JASPER=OFF
    -DWITH_OPENJPEG=OFF
    -DWITH_FFMPEG=OFF
    -DWITH_GSTREAMER=OFF
    -DWITH_OPENCL=OFF
    -DWITH_VTK=OFF
    -DWITH_ADE=OFF
    -DWITH_JPEG=ON
    -DWITH_PNG=ON
    -DWITH_TIFF=ON
)

if(WIN32)
    set(_opencv_debug_suffix "")
    if(_opencv_build_type STREQUAL "Debug")
        set(_opencv_debug_suffix "d")
    endif()
    set(_opencv_ver_header "${PROJECT_SOURCE_DIR}/3rdparty/opencv/modules/core/include/opencv2/core/version.hpp")
    if(EXISTS "${_opencv_ver_header}")
        file(READ "${_opencv_ver_header}" _opencv_ver_text)
        string(REGEX REPLACE ".*#define CV_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _opencv_ver_major "${_opencv_ver_text}")
        string(REGEX REPLACE ".*#define CV_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _opencv_ver_minor "${_opencv_ver_text}")
        string(REGEX REPLACE ".*#define CV_VERSION_REVISION[ \t]+([0-9]+).*" "\\1" _opencv_ver_patch "${_opencv_ver_text}")
        set(_opencv_dllversion "${_opencv_ver_major}${_opencv_ver_minor}${_opencv_ver_patch}")
    else()
        set(_opencv_dllversion "")
    endif()

    # Expose for packaging (used to generate ITLabAIThirdPartyTargets.cmake).
    # Keep it cached so configure_file(@ONLY) in a different directory can read it.
    set(ITLABAI_OPENCV_DLLVERSION "${_opencv_dllversion}" CACHE INTERNAL "OpenCV DLL version suffix for Windows packaging")

    if(MSVC_VERSION GREATER_EQUAL 1930)
        set(_opencv_vc "vc17")
    elseif(MSVC_VERSION GREATER_EQUAL 1920)
        set(_opencv_vc "vc16")
    else()
        set(_opencv_vc "vc15")
    endif()
    set(_opencv_arch "x64")
    set(_opencv_libdir "${OPENCV_INSTALL_DIR}/${_opencv_arch}/${_opencv_vc}/lib")
    set(_opencv_bindir "${OPENCV_INSTALL_DIR}/${_opencv_arch}/${_opencv_vc}/bin")

    set(_opencv_world_lib_release "${_opencv_libdir}/opencv_world${_opencv_dllversion}.lib")
    set(_opencv_world_dll_release "${_opencv_bindir}/opencv_world${_opencv_dllversion}.dll")
    set(_opencv_world_lib_debug "${_opencv_libdir}/opencv_world${_opencv_dllversion}d.lib")
    set(_opencv_world_dll_debug "${_opencv_bindir}/opencv_world${_opencv_dllversion}d.dll")
    set(_opencv_world_lib "${_opencv_libdir}/opencv_world${_opencv_dllversion}${_opencv_debug_suffix}.lib")
    set(_opencv_world_dll "${_opencv_bindir}/opencv_world${_opencv_dllversion}${_opencv_debug_suffix}.dll")
    set(_opencv_include_dir "${OPENCV_INSTALL_DIR}/include")

    set(_opencv_byproducts
        "${_opencv_world_lib}"
        "${_opencv_world_dll}"
    )
else()
    set(_opencv_world "${OPENCV_INSTALL_DIR}/lib/libopencv_world${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_opencv_include_dir "${OPENCV_INSTALL_DIR}/include/opencv4")
    set(ITLABAI_OPENCV_DLLVERSION "" CACHE INTERNAL "OpenCV DLL version suffix for Windows packaging")

    set(_opencv_byproducts
        "${_opencv_world}"
    )
endif()

set(OPENCV_INCLUDE_DIR "${_opencv_include_dir}")

itlabai_external_add(
    NAME opencv_external
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/opencv"
    BINARY_DIR "${OPENCV_BUILD_DIR}"
    INSTALL_DIR "${OPENCV_INSTALL_DIR}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${_opencv_build_type}
        -DOPENCV_INSTALL_BIN_DIR=bin
        -DOPENCV_INSTALL_LIB_DIR=lib
        -DOPENCV_INSTALL_INCLUDE_DIR=include
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PROTOBUF=ON
        -DPROTOBUF_UPDATE_FILES=OFF
        ${OPENCV_BUILD_OPTS}
        ${OPENCV_FEATURE_OPTS}
        ${ITLABAI_EXTERNAL_TOOLCHAIN_ARGS}
        ${ITLABAI_EXTERNAL_WARNING_ARGS_C_AND_CXX}
    CMAKE_CACHE_ARGS
        -DBUILD_LIST:STRING=${OPENCV_COMPONENTS_ESC}
    BUILD_BYPRODUCTS ${_opencv_byproducts}
)

file(MAKE_DIRECTORY "${_opencv_include_dir}")
file(MAKE_DIRECTORY "${OPENCV_INSTALL_DIR}/lib")

add_library(OpenCV::opencv_world SHARED IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(OpenCV::opencv_world PROPERTIES
        IMPORTED_LOCATION "${_opencv_world_dll}"
        IMPORTED_LOCATION_RELEASE "${_opencv_world_dll_release}"
        IMPORTED_LOCATION_DEBUG "${_opencv_world_dll_debug}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_opencv_world_dll_release}"
        IMPORTED_LOCATION_MINSIZEREL "${_opencv_world_dll_release}"
        IMPORTED_IMPLIB "${_opencv_world_lib}"
        IMPORTED_IMPLIB_RELEASE "${_opencv_world_lib_release}"
        IMPORTED_IMPLIB_DEBUG "${_opencv_world_lib_debug}"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_opencv_world_lib_release}"
        IMPORTED_IMPLIB_MINSIZEREL "${_opencv_world_lib_release}"
        INTERFACE_INCLUDE_DIRECTORIES "${_opencv_include_dir}"
    )
else()
    set_target_properties(OpenCV::opencv_world PROPERTIES
        IMPORTED_LOCATION "${_opencv_world}"
        IMPORTED_LOCATION_RELEASE "${_opencv_world}"
        IMPORTED_LOCATION_DEBUG "${_opencv_world}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_opencv_world}"
        IMPORTED_LOCATION_MINSIZEREL "${_opencv_world}"
        INTERFACE_INCLUDE_DIRECTORIES "${_opencv_include_dir}"
    )
endif()

# Ensure OpenCV headers/libs are built/installed before anything that links against it.
add_dependencies(OpenCV::opencv_world opencv_external)
