set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})


unset(TENSORFLOW_FOUND)

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
endif()

if(TENSORFLOW_FOUND)
message(STATUS "TensorFlow_INCLUDE_DIRS: ${tensorflow_INCLUDE_DIRS}")
message(STATUS "TensorFlow_LIBRARIES: ${tensorflow_LIBRARIES}")
else()
message(STATUS "TensorFlow library not found")
endif()
target_include_directories(AlexNetLib PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries( AlexNetLib ${TensorFlow_LIBRARIES} )
