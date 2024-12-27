#include "build.cpp"
#include "build.hpp"

using namespace itlab_2023;

int main() {
  std::string image_path = IMAGE1_PATH;
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    throw std::runtime_error("Failed to load image");
  }
  cv::Mat resized_image;
  cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  cv::resize(image, resized_image, cv::Size(28, 28));
  std::vector<cv::Mat> channels;

  cv::split(resized_image, channels);

  int count_pic = 1;
  std::vector<float> res(count_pic * 28 * 28);
  //
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      res[i * 28 + j] = channels[0].at<uchar>(j, i);
    }
  }
  Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
  // move to reshape layer
  Tensor t = make_tensor<float>(res, sh);
  Tensor input = t;

  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor output = make_tensor(vec, sh1);

  build_graph(input, output);
}
