#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;

TEST(Read_img, can_read_image) {
  const std::string image_path = IMAGE_PATH;
  ASSERT_NO_THROW(Mat image = imread(image_path););
}
<<<<<<< HEAD
=======

>>>>>>> 3367a696f1a0c5e3a22dec439df7612e458500b4
TEST(Read_img, can_save_image) {
  const std::string image_path = IMAGE_PATH;
  Mat image = imread(image_path);
  String output_file_name = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(output_file_name, image););
}
