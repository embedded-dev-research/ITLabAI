file(GLOB_RECURSE SRC_FILES *.cpp)
file(GLOB_RECURSE INCLUDE_HEADERS *.h *.hpp)

set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/alexnet_src.cpp")
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

target_include_directories(AlexNetLib PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries( AlexNetLib ${TensorFlow_LIBRARIES} )
add_executable(alexnet alexnet.cpp)
target_link_libraries(alexnet AlexNetLib)

if (WIN32)
add_custom_command(TARGET alexnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
        "$<TARGET_FILE_DIR:alexnet>")
endif()

file(DOWNLOAD
    "https://tinyurl.com/2w6eh53v"
    "${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite"
     SHOW_PROGRESS
     STATUS status_code
     LOG log_file
)
add_definitions(-DMODEL1_PATH="${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite")
