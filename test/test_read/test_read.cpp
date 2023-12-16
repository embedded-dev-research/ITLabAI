#include "Reader_img.hpp"
#include "gtest/gtest.h"
using namespace cv;
TEST(basic, basic_test) {
  // Arrange
  int a = 2;
  int b = 3;
  // Act
  int c = a + b;
  // Assert
  ASSERT_EQ(5, c);
}

TEST(Read_img, can_read_image) { 
  ASSERT_NO_THROW(Mat image = imread("image.jpg"););
}
TEST(Read_img, can_show_image) {
  Mat image = imread("image.jpg");
  String windowName = "Image";
  namedWindow(windowName);
  imshow(windowName, image);
  ASSERT_NO_THROW(imshow(windowName, image););
}