file(GLOB_RECURSE SRC_FILES *.cpp)
file(GLOB_RECURSE INCLUDE_HEADERS *.h *.hpp)

set(INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.hpp")
set(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/alexnet_src.cpp")
add_library(AlexNetLib STATIC ${INCLUDE_HEADERS} ${SRC_FILES})

if(UNIX)
  unset(TENSORFLOW_FOUND)
  find_path(tensorflow_INCLUDE_DIR
          NAMES
          tensorflow/core
          tensorflow/c
          HINTS
          /usr/local/include/tensorflow
          /usr/include/tensorflow)
  find_library(tensorflow_LIBRARY NAMES tensorflow
          HINTS
          /usr/lib
          /usr/local/lib)
  if(TENSORFLOW_FOUND)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_INCLUDE_DIR tensorflow_LIBRARY)
    set(TensorFlow_LIBRARIES ${tensorflow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${tensorflow_INCLUDE_DIR})
  endif()
endif()

if(WIN32)
    set(TensorFlow_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/include")
    set(TensorFlow_LIBRARIES "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.lib")
    if(EXISTS "${TensorFlow_INCLUDE_DIRS}" AND EXISTS "${TensorFlow_LIBRARIES}")
         include(FindPackageHandleStandardArgs)
         find_package_handle_standard_args(tensorflow DEFAULT_MSG TensorFlow_INCLUDE_DIRS TensorFlow_LIBRARIES)
    endif()
endif()

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
