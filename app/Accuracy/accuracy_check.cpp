#include "acc.hpp"
#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

int main() {
  std::string image_path = IMAGE1_PATH;
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    throw std::runtime_error("Failed to load image");
  }
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(227, 227));
  std::vector<cv::Mat> channels;
  cv::split(resized_image, channels);
  int count_pic = 1;
  std::vector<float> res(count_pic * 227 * 227 * 3);
  int c = 0;
  for (int i = 0; i < 227; ++i) {
    for (int j = 0; j < 227; ++j) {
      res[c] = channels[2].at<uchar>(i, j);
      c++;
      res[c] = channels[1].at<uchar>(i, j);
      c++;
      res[c] = channels[0].at<uchar>(i, j);
      c++;
    }
  }
  Shape sh({static_cast<size_t>(count_pic), 227, 227, 3});
  Tensor t = make_tensor<float>(res, sh);
  Graph graph(6);
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = t;
  Tensor output = make_tensor(vec, sh1);
  auto a1 = std::make_shared<InputLayer>(kNchw, kNchw, 1, 2);
  std::vector<float> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_shared<ConvolutionalLayer>(1, 0, 0, kernel);
  Shape poolshape = {2, 2};
  auto a3 = std::make_shared<EWLayer>("linear", 2.0F, 3.0F);
  auto a4 = std::make_shared<PoolingLayer>(poolshape, "average");
  auto a5 = std::make_shared<OutputLayer>();
  auto a6 = std::make_shared<FCLayer>();
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a3, a4);
  graph.makeConnection(a4, a5);
  graph.makeConnection(a5, a6);
  graph.setOutput(a5, output);
  graph.inference();
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> tmp_output = softmax<float>(*output.as<float>());
  for (float i : tmp) {
    std::cout << i << " ";
  }
}
