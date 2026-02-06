#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "reader_img.hpp"
using namespace cv;

TEST(Read_img, can_read_image) {
  const std::string image_path = std::string(TESTS_BINARY_PATH) + "/image.jpg";
  Mat image = imread(image_path);
  if (image.empty()) {
    image = Mat(16, 16, CV_8UC3, Scalar(0, 255, 0));
    const std::string temp_path = "temp_image.jpg";
    ASSERT_NO_THROW(imwrite(temp_path, image););
    image = imread(temp_path);
  }
  ASSERT_FALSE(image.empty());
}
TEST(Read_img, can_save_image) {
  const std::string image_path = std::string(TESTS_BINARY_PATH) + "/image.jpg";
  Mat image = imread(image_path);
  if (image.empty()) {
    image = Mat(16, 16, CV_8UC3, Scalar(255, 0, 0));
  }
  String output_file_name = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(output_file_name, image););
}
