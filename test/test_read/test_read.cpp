#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;

TEST(Read_img, can_read_image) {
  const std::string image_path = std::string(TESTS_BINARY_PATH) + "/image.jpg";
  ASSERT_NO_THROW(Mat image = imread(image_path););
}

TEST(Read_img, can_save_image) {
  const std::string image_path = std::string(TESTS_BINARY_PATH) + "/image.jpg";
  Mat image = imread(image_path);
  String output_file_name = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(output_file_name, image););
}
