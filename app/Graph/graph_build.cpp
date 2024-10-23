#include "Weights_Reader/reader_weights.hpp"
#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <variant>
#include "build.hpp"

using namespace itlab_2023;

void build_graph(Tensor input, Tensor output) {
  std::vector<std::shared_ptr<Layer>> layers;

  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);

  for (const auto& layer_data : model_data) {

    std::string layer_type = layer_data["type"];
    Tensor tensor = create_tensor_from_json(layer_data["weights"], Type::kFloat);

    if (layer_type.find("Conv") != std::string::npos) {
      Shape shape = tensor.get_shape();
      Tensor tmp_values = make_tensor(tensor.get_values(), shape);
      Tensor tmp_bias = make_tensor(tensor.get_bias());
      layers.push_back(std::make_shared<ConvolutionalLayer>(1, 0, 0, tmp_values, tmp_bias));
    }

    if (layer_type.find("Dense") != std::string::npos) {
      Tensor tmp_values = make_tensor(tensor.get_values(), tensor.get_shape());
      Tensor tmp_bias = make_tensor(tensor.get_bias());
      layers.push_back(std::make_shared<FCLayer>(tmp_values, tmp_bias));
    }

    if (layer_type.find("Pool") != std::string::npos) {
      Shape shape = {2, 2};
      layers.push_back(std::make_shared<PoolingLayer>(shape));
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      layers.emplace_back(/*construcrtor of flatten*/);
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      layers.emplace_back(/*construcrtor of dropout*/);
    }
  }
  Graph graph(layers.size());
  InputLayer a1(kNhwc, kNchw, 1, 2);

  graph.setInput(a1, input);

  graph.makeConnection(a1, *layers[0]);

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    graph.makeConnection(*layers[i], *layers[i + 1]);
  }

  graph.setOutput(*layers.back(), output);

  graph.inference();

  std::vector<float> tmp = *output.as<float>();
  std::vector<float> tmp_output = softmax<float>(*output.as<float>());
  for (float i : tmp) {
    std::cout << i << " ";
  }
}

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