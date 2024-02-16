# Укажем путь к заголовкам вашей библиотеки
set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

# Укажем исходные файлы вашей библиотеки
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

# Создадим цель для вашей библиотеки
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow)

find_library(TensorFlow_LIBRARY NAMES tensorflow_all
        HINTS
        /usr/lib
        /usr/local/lib)

find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
target_link_libraries(AlexNetLib ${TENSORFLOW_LIBRARY})
