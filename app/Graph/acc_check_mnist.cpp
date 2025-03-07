#include <iomanip>
#include <numeric>
#include <sstream>

#include "build.cpp"
#include "build.hpp"

using namespace itlab_2023;

int main() {
  std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                891, 957,  1027, 973,  1008};
  int stat = 0;

  for (size_t name = 0; name < 10; name++) {
    for (size_t ind = 0; ind < counts[name] + 1; ind++) {
      std::ostringstream oss;
      oss << "/" << name << "_" << std::setw(6) << std::setfill('0') << ind
          << ".png";
      std::string png = oss.str();
      std::string image_path = FOLDERMNIST_PATH + png;
      std::cout << image_path << std::endl;

      cv::Mat image = cv::imread(image_path);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image");
      }
      cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      std::vector<cv::Mat> channels;
      cv::split(image, channels);
      int count_pic = 1;
      std::vector<float> res(count_pic * 28 * 28);
      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          res[i * 28 + j] = channels[0].at<uchar>(j, i);
        }
      }
      Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
      Tensor t = make_tensor<float>(res, sh);
      Tensor input = t;
      Shape sh1({1, 5, 5, 3});
      std::vector<float> vec;
      vec.reserve(75);
      for (int i = 0; i < 75; ++i) {
        vec.push_back(3);
      }
      Tensor output = make_tensor(vec, sh1);
      build_graph(input, output, false);
      std::vector<float> tmp_output = softmax<float>(*output.as<float>());
      for (size_t i = 0; i < tmp_output.size(); i++) {
        if (tmp_output[i] >= 1e-6) {
          if (i == name) stat++;
        }
      }
    }
  }

  size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  double percentage =
      (static_cast<double>(stat) / static_cast<double>(sum + 10)) * 100;
  std::cout << "Stat: " << std::fixed << std::setprecision(2) << percentage
            << "%" << std::endl;
  std::cout << percentage << std::endl;
}
