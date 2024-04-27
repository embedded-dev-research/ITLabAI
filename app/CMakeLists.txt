add_subdirectory(ReaderImage)
if(UNIX)
    find_path(tensorflow_INCLUDE_DIR
            NAMES
            tensorflow/core
            tensorflow/c
            HINTS
            /usr/local/include/tensorflow
            /usr/include/tensorflow)
    find_library(tensorflow_LIBRARY NAMES tensorflow
            HINTS
            /usr/lib
            /usr/local/lib)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_INCLUDE_DIR tensorflow_LIBRARY)
    if(TENSORFLOW_FOUND)
        set(TensorFlow_LIBRARIES ${tensorflow_LIBRARY})
        set(TensorFlow_INCLUDE_DIRS ${tensorflow_INCLUDE_DIR})
        SET(TENSORFLOW_FOUND "TRUE" CACHE INTERNAL "TENSORFLOW_FOUND")

    else()
        message(WARNING "TensorFlow not found. Skipping building of alexnet target.")
        SET(TENSORFLOW_FOUND "FALSE" CACHE INTERNAL "TENSORFLOW_FOUND")
        return()
    endif()
endif()

if(WIN32)
    set(TensorFlow_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/include")
    set(TensorFlow_LIBRARIES "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.lib")
    if(EXISTS "${TensorFlow_INCLUDE_DIRS}" AND EXISTS "${TensorFlow_LIBRARIES}")
         include(FindPackageHandleStandardArgs)
         find_package_handle_standard_args(tensorflow DEFAULT_MSG TensorFlow_INCLUDE_DIRS TensorFlow_LIBRARIES)
         SET(TENSORFLOW_FOUND "TRUE" CACHE INTERNAL "TENSORFLOW_FOUND")
    else()
         SET(TENSORFLOW_FOUND "FALSE" CACHE INTERNAL "TENSORFLOW_FOUND")
         return()
    endif()
endif()

if (TENSORFLOW_FOUND)
    add_subdirectory(AlexNet)
else()
    message(WARNING "TensorFlow not found. Skipping building of alexnet target.")
endif()
