#include <iomanip>
#include <numeric>
#include <sstream>

#include "build.cpp"
#include "build.hpp"

using namespace itlab_2023;

int main(int argc, char* argv[]) {
  bool parallel = false;
  if (argc > 1 && std::string(argv[1]) == "--parallel") {
    std::cout << "Parallel mode" << std::endl;
    parallel = true;
  }
  std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                891, 957,  1027, 973,  1008};
  int stat = 0;
  size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  int count_pic = static_cast<int>(sum) + 10;
  std::vector<float> res(count_pic * 28 * 28);
  Tensor input;
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor output = make_tensor(vec, sh1);

  for (size_t name = 0; name < 10; name++) {
    for (size_t ind = 0; ind < counts[name] + 1; ind++) {
      std::ostringstream oss;
      oss << "/" << name << "_" << std::setw(6) << std::setfill('0') << ind
          << ".png";
      std::string png = oss.str();
      std::string image_path = MNIST_PATH + png;

      cv::Mat image = cv::imread(image_path);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image");
      }
      cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      std::vector<cv::Mat> channels;
      cv::split(image, channels);
      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          size_t a = ind;
          for (size_t n = 0; n < name; n++) a += counts[n] + 1;
          res[(a) * 28 * 28 + i * 28 + j] = channels[0].at<uchar>(j, i);
        }
      }
    }
  }
  Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
  Tensor t = make_tensor<float>(res, sh);
  input = t;
  build_graph(input, output, false, parallel);
  std::vector<std::vector<float>> tmp_output =
      softmax<float>(*output.as<float>(), 10);
  std::vector<size_t> indices;
  for (const auto& row : tmp_output) {
    for (size_t j = 0; j < row.size(); ++j) {
      if (row[j] >= 1e-6) {
        indices.push_back(j);
        break;
      }
    }
  }
  for (size_t name = 0; name < 10; name++) {
    for (size_t ind = 0; ind < counts[name] + 1; ind++) {
      size_t a = ind;
      for (size_t n = 0; n < name; n++) a += counts[n] + 1;
      if (name == indices[a]) stat++;
    }
  }
  double percentage =
      (static_cast<double>(stat) / static_cast<double>(sum + 10)) * 100;
  std::cout << "Stat: " << std::fixed << std::setprecision(2) << percentage
            << "%" << std::endl;
}
