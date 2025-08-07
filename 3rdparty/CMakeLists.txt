add_subdirectory(googletest)
add_subdirectory(Json)

set(TBB_TEST OFF CACHE BOOL "Disable TBB tests")
set(TBB_OUTPUT_DIR_BASE tbb CACHE STRING "TBB output directory base")

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_TBB_BACKUP "${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS_TBB_BACKUP "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
endif()

add_subdirectory(TBB)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_TBB_BACKUP}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_TBB_BACKUP}")
endif()

include(ExternalProject)

function(setup_opencv_external_project)
    message(STATUS "Building OpenCV from bundled source")

    set(OPENCV_BINARY_DIR "${CMAKE_BINARY_DIR}/opencv_build")
    set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/opencv_install")
    
    file(MAKE_DIRECTORY "${OPENCV_BINARY_DIR}")
    
    # Create dummy fp.h for macOS compatibility
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include_fix")
    file(WRITE "${CMAKE_BINARY_DIR}/include_fix/fp.h" "#include <math.h>\n")

    set(OPENCV_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_opencv_apps=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
        -DCMAKE_CXX_FLAGS=-w
        -DCMAKE_C_FLAGS=-w
        -DBUILD_opencv_core=ON
        -DBUILD_opencv_imgproc=ON
        -DBUILD_opencv_imgcodecs=ON
        -DBUILD_opencv_highgui=ON
        # Disable problematic modules
        -DBUILD_opencv_video=OFF
        -DBUILD_opencv_videoio=OFF
        -DBUILD_opencv_ml=OFF
        -DBUILD_opencv_features2d=OFF
        -DBUILD_opencv_calib3d=OFF
        -DBUILD_opencv_objdetect=OFF
        -DBUILD_opencv_dnn=OFF
        -DBUILD_opencv_photo=OFF
        -DBUILD_opencv_stitching=OFF
        -DBUILD_opencv_gapi=OFF
        -DBUILD_opencv_python3=OFF
        -DBUILD_opencv_python_bindings_generator=OFF
        # Build image libraries within OpenCV with simplified approach
        -DBUILD_ZLIB=OFF
        -DBUILD_PNG=ON
        -DBUILD_JPEG=ON
        -DBUILD_TIFF=OFF
        -DBUILD_WEBP=OFF
        -DBUILD_OPENEXR=OFF
        # Enable image format support via bundled libraries
        -DWITH_PNG=ON
        -DWITH_JPEG=ON
        -DWITH_TIFF=OFF
        -DWITH_WEBP=OFF
        -DWITH_OPENEXR=OFF
        -DWITH_JASPER=OFF
        -DWITH_OPENJPEG=OFF
        -DBUILD_JASPER=OFF
        -DBUILD_OPENJPEG=OFF
        # Create dummy fp.h and set include path
        -DCMAKE_C_FLAGS=-I${CMAKE_BINARY_DIR}/include_fix
        -DCMAKE_CXX_FLAGS=-I${CMAKE_BINARY_DIR}/include_fix
        # Disable other problematic features
        -DWITH_OPENCL=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENGL=OFF
        -DWITH_QT=OFF
        -DWITH_GTK=OFF
        -DWITH_1394=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_GSTREAMER=OFF
        -DWITH_V4L=OFF
        -DWITH_DSHOW=OFF
        -DWITH_EIGEN=OFF
        -DWITH_LAPACK=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_DOCS=OFF
        -DBUILD_PACKAGE=OFF
        # Force use of bundled 3rdparty libraries
        -DOPENCV_FORCE_3RDPARTY_BUILD=ON
        # macOS specific fixes
        -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
        -DCMAKE_CXX_STANDARD=17
        -DWITH_IPP=OFF
        -DWITH_ITT=OFF
        # Disable iOS/mobile features
        -DWITH_OPENVX=OFF
        -DBUILD_ANDROID_PROJECTS=OFF
        -DBUILD_ANDROID_EXAMPLES=OFF
        # Disable ARM optimizations that cause linker issues
        -DWITH_CAROTENE=OFF
        -DCPU_BASELINE=""
        -DCPU_DISPATCH=""
        # Use system ZLIB (macOS has it built-in)
        -DWITH_ZLIB=ON
    )

    if(CMAKE_C_COMPILER_LAUNCHER)
        list(APPEND OPENCV_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER})
    endif()

    if(CMAKE_CXX_COMPILER_LAUNCHER)
        list(APPEND OPENCV_CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
    
    # Add generator if present
    set(OPENCV_GENERATOR_ARGS "")
    if(CMAKE_GENERATOR)
        set(OPENCV_GENERATOR_ARGS -G ${CMAKE_GENERATOR})
    endif()

    ExternalProject_Add(
        opencv_external
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv"
        BINARY_DIR "${OPENCV_BINARY_DIR}"
        CMAKE_ARGS ${OPENCV_CMAKE_ARGS}
        BUILD_ALWAYS OFF
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}/3rdparty/opencv" -B "${OPENCV_BINARY_DIR}" ${OPENCV_GENERATOR_ARGS} ${OPENCV_CMAKE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build "${OPENCV_BINARY_DIR}" --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build "${OPENCV_BINARY_DIR}" --target install
    )

    add_library(opencv_interface INTERFACE)
    add_dependencies(opencv_interface opencv_external)

    target_include_directories(opencv_interface INTERFACE
        ${OPENCV_INSTALL_DIR}/include/opencv4
        ${OPENCV_INSTALL_DIR}/include
    )

    target_link_directories(opencv_interface INTERFACE
        ${OPENCV_INSTALL_DIR}/lib
        ${OPENCV_INSTALL_DIR}/lib/opencv4/3rdparty
    )

    target_link_libraries(opencv_interface INTERFACE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        # Link bundled PNG and JPEG libraries
        libpng
        libjpeg-turbo
        # Link system ZLIB for PNG compression
        z
    )
    
    # Add essential macOS frameworks for OpenCV highgui (GUI functionality)
    if(APPLE)
        target_link_libraries(opencv_interface INTERFACE
            "-framework Cocoa"
            "-framework Foundation"
        )
    endif()

endfunction()

setup_opencv_external_project()
