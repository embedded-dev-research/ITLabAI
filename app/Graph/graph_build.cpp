#include "build.cpp"
#include "build.hpp"

namespace fs = std::filesystem;
using namespace it_lab_ai;

int main(int argc, char* argv[]) {
  std::string image_folder = IMAGE1_PATH;
  std::vector<std::string> image_paths;
  bool parallel = false;
  if (argc > 1 && std::string(argv[1]) == "--parallel") {
    std::cout << "Parallel mode" << std::endl;
    parallel = true;
  }

  for (const auto& entry : fs::directory_iterator(image_folder)) {
    if (entry.path().extension() == ".png") {
      image_paths.push_back(entry.path().string());
    }
  }

  if (image_paths.empty()) {
    throw std::runtime_error("No PNG images found in the folder");
  }

  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Failed to load image: " << image_path << std::endl;
      continue;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    std::vector<float> res(28 * 28);
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        res[i * 28 + j] = channels[0].at<uchar>(j, i);
      }
    }

    Shape sh({1, 1, 28, 28});
    Tensor input = make_tensor<float>(res, sh);

    Shape sh1({1, 5, 5, 3});
    std::vector<float> vec(75, 3);
    Tensor output = make_tensor(vec, sh1);

    build_graph(input, output, true, parallel);

    std::vector<float> tmp_output = softmax<float>(*output.as<float>());
    for (size_t i = 0; i < tmp_output.size(); i++) {
      if (tmp_output[i] >= 1e-6) {
        std::cout << "Image: " << image_path << " -> Class: " << i << std::endl;
      }
    }
  }
  return 0;
}
