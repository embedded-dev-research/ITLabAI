#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;

TEST(Read_img, can_read_image) {
  const std::string imagePath = IMAGE_PATH;
  ASSERT_NO_THROW(Mat image = imread(imagePath););
}
TEST(Read_img, can_save_image) {
  const std::string imagePath = IMAGE_PATH;
  Mat image = imread(imagePath);
  String output_file_name = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(output_file_name, image););
}
