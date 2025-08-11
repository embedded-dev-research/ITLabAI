#include <stdexcept>
#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

TEST(bfs, check_result_vec) {
  Graph graph(5);
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  InputLayer a3(kNhwc, kNhwc, 1, 1);
  a1.setName(kInput);
  a3.setName(kInput);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a4(1, 0, 1, kernel);
  a2.setName(kConvolution);
  a4.setName(kConvolution);
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a1, a3);
  graph.makeConnection(a2, a4);
  graph.setOutput(a4, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {81, 81, 81};
#ifdef ENABLE_STATISTIC_TENSORS
  std::vector<Tensor> tensors = graph.getTensors();
  for (size_t i = 0; i < tensors.size(); i++) {
    std::vector<int> ten = *tensors[i].as<int>();
    for (size_t j = 0; j < ten.size(); j++) {
      std::cout << ten[j] << ' ';
    }
    std::cout << '\n';
  }
#endif
#ifdef ENABLE_STATISTIC_TIME
  std::vector<std::string> times = graph.getTimeInfo();
  for (size_t j = 0; j < times.size(); j++) {
    std::cout << times[j] << ' ';
  }
  std::cout << '\n';
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> weights = graph.getWEIGHTS();
  for (size_t i = 0; i < weights.size(); i++) {
    switch (weights[i].get_type()) {
      case Type::kInt: {
        std::vector<int> ten = *weights[i].as<int>();
        for (size_t j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kFloat: {
        std::vector<float> ten = *weights[i].as<float>();
        for (size_t j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kUnknown:
      default: {
        throw std::runtime_error("Unknown tensor type encountered");
        break;
      }
    }
  }
#endif
  ASSERT_EQ(tmp, res);
}
TEST(bfs, check_end_to_end) {
  Graph graph(6);
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  std::vector<float> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  Shape poolshape = {2, 2};
  EWLayer a3("linear", 2.0F, 3.0F);
  PoolingLayer a4(poolshape, "average");
  FCLayer a6;
  OutputLayer a5;
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a3, a4);
  graph.makeConnection(a4, a5);
  graph.makeConnection(a5, a6);
  graph.setOutput(a5, output);
  graph.inference();
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> weights = graph.getWEIGHTS();
  for (size_t i = 0; i < weights.size(); i++) {
    switch (weights[i].get_type()) {
      case Type::kInt: {
        std::vector<int> ten = *weights[i].as<int>();
        for (size_t j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kFloat: {
        std::vector<float> ten = *weights[i].as<float>();
        for (size_t j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kUnknown:
      default: {
        throw std::runtime_error("Unknown tensor type encountered");
        break;
      }
    }
  }
#endif
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> tmp_output = softmax<float>(*output.as<float>());
  std::vector<float> res(3, 21);
  ASSERT_EQ(tmp, res);
}
TEST(bfs, check_struct_layer) {
  Graph graph(5);
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  a1.setName(kInput);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a3(1, 0, 1, kernel);

  // EWLayer a4("linear", 2.0F, 3.0F);
  // a2.ewops.layers.push_back(&a4);
  // a2.ewops.countlayers++;

  a2.setName(kConvolution);
  a3.setName(kConvolution);
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.setOutput(a3, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {81, 81, 81};
  ASSERT_EQ(tmp, res);
}
TEST(bfs, check_struct_layer_added) {
  Graph graph(5);
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  a1.setName(kInput);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a3(1, 0, 1, kernel);

  EWLayer a4("linear", 2.0F, 3.0F);
  a2.postops.layers.push_back(&a4);
  a2.postops.count++;

  a2.setName(kConvolution);
  a3.setName(kConvolution);
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.setOutput(a3, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {189, 189, 189};
  ASSERT_EQ(tmp, res);
}
