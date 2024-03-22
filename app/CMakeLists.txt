add_executable(Reader reader_img.cpp)
add_executable(alexnet alexnet.cpp)
target_link_libraries(alexnet AlexNetLib)
target_link_libraries(Reader ReadLib)

if (WIN32)
add_custom_command(TARGET Reader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}"
        "${CMAKE_BINARY_DIR}/app/${CMAKE_BUILD_TYPE}/")
add_custom_command(TARGET alexnet POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
            "$<TARGET_FILE_DIR:alexnet>")
endif()
file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DIMAGE1_PATH="${CMAKE_CURRENT_BINARY_DIR}/image.jpg")
file(DOWNLOAD
    "https://storage.googleapis.com/kagglesdsdata/models/2508/3404/saved_model.pb?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240315%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240315T140446Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3fab9972bcbe0411f73d9a10fe78ba84ce3d97c1f440c79e0fb388b36775170375b272cd833f76259fba9416d2ab09eedcfc69815c95c32a45ec35d61bd647952b27e68bc7bf8e2283ea7465d5fcbf0a85d2f91af5ea1ea8a07ad19ba4d82f15b363489f69feb6ce22809ac7b1edd877fd9b9162e678d83fef9b8f0b2b374f6b6eaadaa45a83f8462c5182804dc89d87dc1d368b182bcc512a39263fb907c93d76a7a8e4f4b10eba23c8ac733e228b94ffb3a36b8bff389b4a06cdeeae89011f4665f88ee6943ae8aacf1276a8e6291007d29891ef5cd75d01adc4a1b3757450a032e38bfc5eb16756b30e23ec40109e2825765f9ec45e3475181c77f96afbc7"
    "${CMAKE_CURRENT_BINARY_DIR}/saved_model.pb"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
add_definitions(-DMODEL1_PATH="${CMAKE_CURRENT_BINARY_DIR}/saved_model.pb")
