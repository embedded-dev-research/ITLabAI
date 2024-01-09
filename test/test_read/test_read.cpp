#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;
TEST(Read_img, can_read_image) {
  ASSERT_NO_THROW(Mat image = imread("../include/read/image.jpg"););
}
TEST(Read_img, can_show_image) {
  Mat image = imread("../include/read/image.jpg");
  String windowName = "Image";
  namedWindow(windowName);
  imshow(windowName, image);
  ASSERT_NO_THROW(imshow(windowName, image););
}
