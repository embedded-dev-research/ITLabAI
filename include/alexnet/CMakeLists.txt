set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/c
        HINTS
        /usr/local/include/tensorflow
        /usr/include/tensorflow)

find_library(TensorFlow_LIBRARY NAMES tensorflow
        HINTS
        /usr/lib
        /usr/local/lib)


find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

target_link_libraries( AlexNetLib ${TensorFlow_LIBRARIES} )
