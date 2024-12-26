#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using namespace itlab_2023;

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
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 0, kernel);
  ConvolutionalLayer a4(1, 0, 0, kernel);
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
  for (int i = 0; i < tensors.size(); i++) {
    std::vector<int> ten = *tensors[i].as<int>();
    for (int j = 0; j < ten.size(); j++) {
      std::cout << ten[j] << ' ';
    }
    std::cout << '\n';
  }
#endif
#ifdef ENABLE_STATISTIC_TIME
  std::vector<int> times = graph.getTime();
  for (int j = 0; j < times.size(); j++) {
    std::cout << times[j] << ' ';
  }
  std::cout << '\n';
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> weights = graph.getWEIGHTS();
  for (int i = 0; i < weights.size(); i++) {
    switch (weights[i].get_type()) {
      case Type::kInt: {
        std::vector<int> ten = *weights[i].as<int>();
        for (int j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kFloat: {
        std::vector<float> ten = *weights[i].as<float>();
        for (int j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
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
  ConvolutionalLayer a2(1, 0, 0, kernel);
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
  for (int i = 0; i < weights.size(); i++) {
    switch (weights[i].get_type()) {
      case Type::kInt: {
        std::vector<int> ten = *weights[i].as<int>();
        for (int j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
        break;
      }
      case Type::kFloat: {
        std::vector<float> ten = *weights[i].as<float>();
        for (int j = 0; j < ten.size(); j++) {
          std::cout << ten[j] << ' ';
        }
        std::cout << '\n';
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
