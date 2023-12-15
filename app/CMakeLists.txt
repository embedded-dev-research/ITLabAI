add_executable(Reader reader_img.cpp)

target_include_directories(Reader PRIVATE ../include)

target_link_libraries(Reader MyInclude)
