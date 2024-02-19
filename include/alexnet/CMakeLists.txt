set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/c
        third_party
        HINTS
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow)

find_library(TensorFlow_LIBRARY NAMES tensorflow_all
        HINTS
        /usr/lib
        /usr/local/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

#set_target_properties(AlexNetLib PROPERTIES LINKER_LANGUAGE CXX)

#find_package( TensorFlow REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libtensorflow-cpu-linux-x86_64-2.7.0/include/TensorFlow)
#include_directories( ${TensorFlow_INCLUDE_DIRS} )
#target_link_libraries( AlexNetLib ${TensorFlow_LIBS} )
