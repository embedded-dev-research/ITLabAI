#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>

#include "Weights_Reader/reader_weights.hpp"
#include "build.hpp"
#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/DropOutLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using namespace itlab_2023;

void build_graph(Tensor input, Tensor output) {
  std::vector<std::shared_ptr<Layer>> layers;

  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);

  std::cout << "Loaded model data from JSON." << std::endl;

  for (const auto& layer_data : model_data) {
    std::string layer_type = layer_data["type"];
    std::cout << "Processing layer of type: " << layer_type << std::endl;

    Tensor tensor =
        create_tensor_from_json(layer_data["weights"], Type::kFloat);

    if (layer_type.find("Conv") != std::string::npos) {
      Shape shape = tensor.get_shape();
      std::cout << "PoolingLayer shape: ";
      for (size_t i = 0; i < shape.dims(); ++i) {
        std::cout << shape[i] << " ";
      }
      std::cout << std::endl;

      Tensor tmp_values = tensor;
      Tensor tmp_bias = make_tensor(tensor.get_bias());

      auto conv_layer =
          std::make_shared<ConvolutionalLayer>(1, 0, 0, tmp_values, tmp_bias);
      conv_layer->setName(kConvolution);
      layers.push_back(conv_layer);
      std::cout << "ConvLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dense") != std::string::npos) {
      Tensor tmp_values = tensor;
      Tensor tmp_bias = make_tensor(tensor.get_bias());

      auto fc_layer = std::make_shared<FCLayer>(tmp_values, tmp_bias);
      fc_layer->setName(kFullyConnected);
      layers.push_back(fc_layer);
      std::cout << "DenseLayer added to layers." << std::endl;
    }

    if (layer_type.find("Pool") != std::string::npos) {
      Shape shape = {2, 2};
      std::cout << "PoolingLayer shape: " << shape[0] << "x" << shape[1]
                << std::endl;
      auto pool_layer = std::make_shared<PoolingLayer>(shape);
      pool_layer->setName(kPooling);
      layers.push_back(pool_layer);
      std::cout << "PoolingLayer added to layers." << std::endl;
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      auto flatten_layer = std::make_shared<FlattenLayer>();
      flatten_layer->setName(kFlatten);
      layers.push_back(flatten_layer);
      std::cout << "FlattenLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      auto dropout_layer = std::make_shared<DropOutLayer>(0.5);
      dropout_layer->setName(kDropout);
      layers.push_back(dropout_layer);
      std::cout << "DropOutLayer added to layers with probability 0.5."
                << std::endl;
    }
  }
  std::cout << "number of layers - " << layers.size() + 1<< std::endl;
  Graph graph(static_cast<int>(layers.size()));
  InputLayer a1(kNhwc, kNchw, 1, 2);

  std::cout << "InputLayer created." << std::endl;

  graph.setInput(a1, input);
  std::cout << "Input set in graph." << std::endl;

  graph.makeConnection(a1, *layers[0]);
  std::cout << "Connection made between InputLayer and first layer."
            << std::endl;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    graph.makeConnection(*layers[i], *layers[i + 1]);
    std::cout << "Connection made between layer " << i << " ("
              << layerTypeToString(layers[i]->getName()) << ")"
              << " and layer " << i + 1 << " ("
              << layerTypeToString(layers[i + 1]->getName()) << ")"
              << std::endl;
  }



  graph.setOutput(*layers.back(), output);
  std::cout << "Output set in graph." << std::endl;

  std::cout << "Starting inference..." << std::endl;
  graph.inference();
  std::cout << "Inference completed." << std::endl;

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
