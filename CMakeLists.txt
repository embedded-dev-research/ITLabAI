cmake_minimum_required(VERSION 3.20)

project(itlab_2023)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

include_directories("include")

add_subdirectory(3rdparty)

if (NOT WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /wd4996 /WX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /wd4996 /WX")
endif()

add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
