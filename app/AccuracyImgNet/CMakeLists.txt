add_executable(ACCImgNet accimgnet.cpp)

target_link_libraries( ACCImgNet opencv_interface )
target_link_libraries( ACCImgNet tbb)
target_link_libraries( ACCImgNet layers_lib)
target_link_libraries( ACCImgNet gtest_main)
