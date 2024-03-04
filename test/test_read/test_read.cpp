#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "read/reader_img.hpp"
using namespace cv;

std::string getImagePath() {
  char* workspace = getenv("GITHUB_WORKSPACE");
  if (workspace != nullptr) {
    return std::string(workspace) + "/build/test/image.jpg";
  }
  return "../image.jpg";
}
TEST(Read_img, can_read_image) {
  ASSERT_NO_THROW(Mat image = imread(getImagePath()););
}
TEST(Read_img, can_save_image1) {
  Mat image = imread(getImagePath());
  String output_file_name = "output_image.jpg";
  ASSERT_NO_THROW(imwrite(output_file_name, image););
}
