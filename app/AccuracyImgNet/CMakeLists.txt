add_executable(ACCImgNet accimgnet.cpp)

find_package( OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ACCImgNet ${OpenCV_LIBS} )
target_link_libraries( ACCImgNet TBB::tbb)
target_link_libraries( ACCImgNet layers_lib)
target_link_libraries( ACCImgNet gtest_main)
