set(OPENCV_PREFIX "${ITLABAI_EXTERNAL_ROOT}/opencv")
set(OPENCV_BUILD_DIR "${ITLABAI_EXTERNAL_BUILD_ROOT}/opencv_min")
set(OPENCV_INSTALL_DIR "${ITLABAI_EXTERNAL_INSTALL_ROOT}/opencv_min")

# Always build local OpenCV (no system fallback for reproducibility)
set(OpenCV_FOUND FALSE)

if(NOT OpenCV_FOUND)
    set(OPENCV_COMPONENTS core imgproc imgcodecs highgui world)
    set(OPENCV_COMPONENTS_ESC "core\\;imgproc\\;imgcodecs\\;highgui\\;world")
    set(OPENCV_FEATURE_ARGS
        -DBUILD_TESTS=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_DOCS=OFF
        -DBUILD_opencv_apps=OFF
        -DBUILD_opencv_dnn=OFF
        -DBUILD_opencv_python=OFF
        -DBUILD_JAVA=OFF
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
        -DBUILD_opencv_world=ON
        -DOPENCV_ENABLE_NONFREE=OFF
        -DOPENCV_DOWNLOAD_PATH=${CMAKE_SOURCE_DIR}/3rdparty/opencv_downloads
        -DOPENCV_DISABLE_ADE=ON
        -DINSTALL_C_EXAMPLES=OFF
        -DINSTALL_PYTHON_EXAMPLES=OFF
        -DBUILD_IPP_IW=OFF
        -DOPENCV_GENERATE_PKGCONFIG=OFF
        -DOPENCV_INSTALL_FFMPEG=OFF
        -DINSTALL_CASCADES=OFF
        -DINSTALL_TESTS=OFF
        -DCMAKE_CXX_STANDARD=17
    )

    ExternalProject_Add(opencv_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv"
        BINARY_DIR "${OPENCV_BUILD_DIR}"
        INSTALL_DIR "${OPENCV_INSTALL_DIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=ON
            -DBUILD_PROTOBUF=ON
            -DPROTOBUF_UPDATE_FILES=OFF
            ${OPENCV_FEATURE_ARGS}
        CMAKE_CACHE_ARGS
            -DBUILD_LIST:STRING=${OPENCV_COMPONENTS_ESC}
        BUILD_BYPRODUCTS
            ${OPENCV_INSTALL_DIR}/lib/libopencv_world${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    add_dependencies(itlabai_external opencv_external)

    if(MSVC)
        set(_opencv_world "${OPENCV_INSTALL_DIR}/lib/opencv_world.lib")
    else()
        set(_opencv_world "${OPENCV_INSTALL_DIR}/lib/libopencv_world${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    file(MAKE_DIRECTORY "${OPENCV_INSTALL_DIR}/include/opencv4")
    file(MAKE_DIRECTORY "${OPENCV_INSTALL_DIR}/lib")

    add_library(OpenCV::opencv_world SHARED IMPORTED GLOBAL)
    set_target_properties(OpenCV::opencv_world PROPERTIES
        IMPORTED_LOCATION "${_opencv_world}"
        IMPORTED_LOCATION_RELEASE "${_opencv_world}"
        IMPORTED_LOCATION_DEBUG "${_opencv_world}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${_opencv_world}"
        IMPORTED_LOCATION_MINSIZEREL "${_opencv_world}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_INSTALL_DIR}/include/opencv4"
    )
else()
    # System OpenCV: ensure a world-like target exists
    if(NOT TARGET OpenCV::opencv_world)
        add_library(OpenCV::opencv_world INTERFACE IMPORTED)
        if(TARGET OpenCV::opencv_core)
            target_link_libraries(OpenCV::opencv_world INTERFACE OpenCV::opencv_core)
        endif()
        if(TARGET OpenCV::opencv_imgproc)
            target_link_libraries(OpenCV::opencv_world INTERFACE OpenCV::opencv_imgproc)
        endif()
        if(TARGET OpenCV::opencv_imgcodecs)
            target_link_libraries(OpenCV::opencv_world INTERFACE OpenCV::opencv_imgcodecs)
        endif()
        if(TARGET OpenCV::opencv_highgui)
            target_link_libraries(OpenCV::opencv_world INTERFACE OpenCV::opencv_highgui)
        endif()
        if(OpenCV_INCLUDE_DIRS)
            set_target_properties(OpenCV::opencv_world PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        endif()
    endif()
endif()
