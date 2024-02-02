

# Найдем пакет TensorFlow
find_package(TensorFlow REQUIRED)

set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

# Включим заголовочные файлы TensorFlow
target_include_directories(alexnet PRIVATE ${TensorFlow_INCLUDE_DIRS})

# Линковка с библиотеками TensorFlow
target_link_libraries(alexnet PRIVATE ${TensorFlow_LIBRARIES})

set_property(TARGET alexnet PROPERTY CXX_STANDARD 11)
