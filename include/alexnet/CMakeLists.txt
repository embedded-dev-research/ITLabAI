# Найдем пакет TensorFlow
set(TENSORFLOW_LIB_DIR "/home/osboxes/.local/lib/python3.6/site-packages/tensorflow")

set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

link_directories(${TENSORFLOW_LIB_DIR})
target_link_libraries(AlexNetLib tensorflow)

