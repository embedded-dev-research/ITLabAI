cmake_minimum_required(VERSION 3.20)

project(ITLabAI)

option(ENABLE_STATISTIC_TENSORS "Enable statistic tensors" OFF)

if(ENABLE_STATISTIC_TENSORS)
    add_definitions(-DENABLE_STATISTIC_TENSORS)
endif()

option(ENABLE_STATISTIC_TIME "Enable statistic time" OFF)

if(ENABLE_STATISTIC_TIME)
    add_definitions(-DENABLE_STATISTIC_TIME)
endif()

option(ENABLE_STATISTIC_WEIGHTS "Enable statistic weights" OFF)

if(ENABLE_STATISTIC_WEIGHTS)
    add_definitions(-DENABLE_STATISTIC_WEIGHTS)
endif()

set(CMAKE_CXX_STANDARD 17)

enable_testing()

include_directories("include")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(3rdparty)

include(cmake/opencv_config.cmake)

if (NOT WIN32)
    find_package(OpenMP REQUIRED)
endif()

if (NOT WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /wd4996 /wd4190 /wd4189 /WX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /wd4996 /wd4190 /wd4189 /WX")
endif()

foreach(CONFIG "" _DEBUG _RELEASE)
    set("CMAKE_ARCHIVE_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/lib")
    set("CMAKE_LIBRARY_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/lib")
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY${CONFIG}" "${CMAKE_BINARY_DIR}/bin")
endforeach()

add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
