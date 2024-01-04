#include "reader_img.hpp"
#include <stdexcept>
using namespace cv;
void read(std::string path) {
  Mat image = imread(path);
  if (image.empty()) {
    throw std::runtime_error("Could not open or find the image");
    cin.get();
  }
  String windowName = "Image";
  namedWindow(windowName);
  imshow(windowName, image);
  waitKey(0);
  destroyWindow(windowName);
}