#include <stdexcept>

#include "reader_img.hpp"
using namespace cv;
void read(std::string& path) {
  Mat image = imread(path);
  if (image.empty()) {
    throw std::runtime_error("Could not open or find the image");
  }
  String window_name = "Image";
  namedWindow(window_name);
  imshow(window_name, image);
  waitKey(0);
  destroyWindow(window_name);
}
