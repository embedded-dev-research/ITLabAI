#include "read/reader_img.hpp"
#include <stdexcept>
using namespace cv;
void read(std::string path) {
  Mat image = imread(path);
  if (image.empty()) {
    throw std::runtime_error("Could not open or find the image");
    std::cin.get();
  }
  String window_name = "Image";
  namedWindow(window_name);
  imshow(window_name, image);
  waitKey(0);
  destroyWindow(window_name);
}