set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set_target_properties(AlexNetLib PROPERTIES LINKER_LANGUAGE CXX)

find_package( TensorFlow REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libtensorflow-cpu-linux-x86_64-2.7.0/include/TensorFlow)
include_directories( ${TensorFlow_INCLUDE_DIRS} )
target_link_libraries( AlexNetLib ${TensorFlow_LIBS} )
