# Укажем путь к заголовкам вашей библиотеки
set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")

# Укажем исходные файлы вашей библиотеки
set(SRC_FILES "${CMAKE_SOURCE_DIR}/src/alexnet/alexnet_src.cpp")

# Создадим цель для вашей библиотеки
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

set(TENSORFLOW_ROOT "/home/osboxes/.local/lib/python3.6/site-packages/tensorflow")

find_library(TENSORFLOW_LIBRARY
NAMES tensorflow
PATHS ${TENSORFLOW_ROOT}
)

target_link_libraries(AlexNetLib ${TENSORFLOW_LIBRARY})
