# Найдем пакет TensorFlow
set(TENSORFLOW_LIB_DIR "/home/osboxes/.local/lib/python3.6/site-packages")

# Укажем путь к заголовкам вашей библиотеки
set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

# Укажем исходные файлы вашей библиотеки
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

# Создадим цель для вашей библиотеки
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

# Установим путь поиска для библиотек TensorFlow
set(CMAKE_PREFIX_PATH ${TENSORFLOW_LIB_DIR})

# Попытаемся найти библиотеку TensorFlow
find_library(TENSORFLOW_LIBRARY tensorflow HINTS ${TENSORFLOW_LIB_DIR})

# Проверим, найдена ли библиотека TensorFlow
if(NOT TENSORFLOW_LIBRARY)
message(FATAL_ERROR "Не удалось найти библиотеку TensorFlow")
endif()

# Свяжем библиотеку TensorFlow с вашей целью
target_link_libraries(AlexNetLib ${TENSORFLOW_LIBRARY})
