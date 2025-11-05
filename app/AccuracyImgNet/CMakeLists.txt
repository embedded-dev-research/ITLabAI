add_executable(ACCImgNet accimgnet.cpp)

find_package( OpenCV REQUIRED PATHS "${OPENCV_BUILD_DIR}" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ACCImgNet ${OpenCV_LIBS} )
target_link_libraries( ACCImgNet TBB_unified)
target_link_libraries( ACCImgNet layers_lib)
target_link_libraries( ACCImgNet gtest_main)
