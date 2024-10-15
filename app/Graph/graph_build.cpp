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
#include "build.hpp"

using namespace itlab_2023;

void build_graph(Tensor input) {

  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);


  int number_of_layers;
  ConvolutionalLayer conv1;
  ConvolutionalLayer conv2;
  ConvolutionalLayer conv3;
  ConvolutionalLayer conv4;
  ConvolutionalLayer conv5;
  FCLayer fc1;
  FCLayer fc2;
  FCLayer fc1;
  FCLayer fc4;


  for (auto& layer : model_data.items()) {
    number_of_layers++;
    std::string layer_name = layer.key();
    Tensor tensor = create_tensor_from_json(layer.value(), Type::kFloat);
    if (layer_name == "layer_conv_1") {
      ConvolutionalLayer conv1(1, 0, 0, tensor);
    }
    if (layer_name == "layer_conv_2") {
      ConvolutionalLayer conv2(1, 0, 0, tensor);
    }
    if (layer_name == "layer_conv_3") {
      ConvolutionalLayer conv3(1, 0, 0, tensor);
    }
    if (layer_name == "layer_conv_4") {
      ConvolutionalLayer conv4(1, 0, 0, tensor);
    }
    if (layer_name == "layer_conv_5") {
      ConvolutionalLayer conv5(1, 0, 0, tensor);
    }
    if (layer_name == "dense") {
      FCLayer fc1 (tensor, tensor.get_bias());
    }
    if (layer_name == "dense_1") {
      FCLayer fc2(tensor, tensor.get_bias());
    }
    if (layer_name == "dense_2") {
      FCLayer fc3(tensor, tensor.get_bias());
    }
    if (layer_name == "layer_conv_4") {
      FCLayer fc4(tensor, tensor.get_bias());
    }
  }


  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Graph graph(number_of_layers);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  Shape poolshape = {2, 2};
  //EWLayer a3("linear", 2.0F, 3.0F);
  PoolingLayer pool1(poolshape, "average");
  PoolingLayer pool2(poolshape, "average");
  PoolingLayer pool3(poolshape, "average");
  PoolingLayer pool4(poolshape, "average");
  OutputLayer output;


  graph.setInput(a1, input);
  graph.makeConnection(a1, conv1);
  graph.makeConnection(conv1, pool1);
  graph.makeConnection(pool1, conv2);
  graph.makeConnection(conv2, pool2);
  graph.makeConnection(pool2, conv3);
  graph.makeConnection(conv3, conv4);
  graph.makeConnection(conv4, conv5);
  graph.makeConnection(conv5, pool3);
  graph.makeConnection(pool3, /* flatten1*/);
  graph.makeConnection(/* flatten1*/, fc1);
  graph.makeConnection(fc1, fc2);
  graph.makeConnection(fc2, /*dropout*/);
  graph.setOutput(/*dropout*/, output);
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
  Tensor input = t; ////////////////////

  
  build_graph(t);
}