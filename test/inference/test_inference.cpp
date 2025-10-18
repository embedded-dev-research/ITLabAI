#include <stdexcept>
#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/BinaryOpLayer.hpp"
#include "layers/ConcatLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "layers/SplitLayer.hpp"
#include "utils/flaky_test_runner.hpp"

using namespace it_lab_ai;

TEST(bfs, check_struct_graph) {
  Graph graph(151);
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);

  ConvolutionalLayer a3_1(1, 0, 1, kernel);
  EWLayer a3_1_1("relu");
  ConvolutionalLayer a3_2(1, 0, 1, kernel);
  EWLayer a3_2_1("relu");

  ConcatLayer a4(0);
  EWLayer a5("relu");

  EWLayer a6_1("relu");
  EWLayer a6_2("relu");

  ConcatLayer a7(0);
  // EWLayer a8("relu");
  SplitLayer a8(1, 3);

  EWLayer a9_1("relu");
  EWLayer a9_2("relu");
  EWLayer a9_3("relu");

  ConcatLayer a10(0);
  EWLayer a11_1("relu");

  ConcatLayer a12(0);

  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3_1);
  graph.makeConnection(a2, a3_2);
  graph.makeConnection(a3_1, a3_1_1);
  graph.makeConnection(a3_1_1, a4);
  graph.makeConnection(a3_2, a3_2_1);
  graph.makeConnection(a3_2_1, a4);
  graph.makeConnection(a4, a5);
  graph.makeConnection(a5, a7);
  graph.makeConnection(a5, a6_1);
  graph.makeConnection(a5, a6_2);
  graph.makeConnection(a6_1, a7);
  graph.makeConnection(a6_2, a7);
  graph.makeConnection(a7, a8);
  graph.makeConnection(a8, a9_1);
  graph.makeConnection(a8, a9_2);
  graph.makeConnection(a8, a9_3);
  graph.makeConnection(a9_1, a10);
  graph.makeConnection(a9_2, a10);
  graph.makeConnection(a9_3, a10);
  graph.makeConnection(a10, a11_1);
  graph.makeConnection(a11_1, a12);
  graph.makeConnection(a10, a12);
  graph.setOutput(a12, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(36, 81);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_not_used_yolo) {
  Graph graph(151);
  Shape sh1({1, 4, 2, 2});
  std::vector<int> vec;
  vec.reserve(16);
  for (int i = 0; i < 16; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  // EWLayer a2("relu"); //split 1 , 4
  SplitLayer a2(1, 4);

  EWLayer a3_1("relu");
  EWLayer a3_1_1("relu");

  ConcatLayer a3_2(0);
  EWLayer a3_2_1("relu");

  EWLayer a3_3("relu");
  ConcatLayer a3_3_1(0);
  EWLayer a3_3_2("relu");
  EWLayer a3_3_3("relu");
  EWLayer a3_3_4("relu");

  ConcatLayer a4(0);

  graph.setInput(a2, input);
  graph.makeConnection(a2, a3_1);
  graph.makeConnection(a2, a3_2);
  graph.makeConnection(a2, a3_3);
  graph.makeConnection(a3_1, a3_1_1);
  graph.makeConnection(a3_1_1, a4);
  graph.makeConnection(a3_2, a3_2_1);
  graph.makeConnection(a3_2_1, a4);
  graph.makeConnection(a3_3, a3_3_1);
  graph.makeConnection(a2, a3_3_1);
  graph.makeConnection(a3_3_1, a3_3_2);
  graph.makeConnection(a3_3_2, a3_3_3);
  graph.makeConnection(a3_3_3, a3_3_4);
  graph.makeConnection(a3_3_4, a3_2);

  graph.setOutput(a4, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(16, 3);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_resnet1) {
  Graph graph(151);
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  SplitLayer a2(1, 2);

  EWLayer a2_1("relu");
  EWLayer a2_1_1("relu");

  EWLayer a2_1_1_1("relu");
  EWLayer a2_1_1_2("relu");

  BinaryOpLayer a2_1_2(BinaryOpLayer::Operation::kMul);
  EWLayer a2_1_3("relu");
  EWLayer a2_2("relu");
  BinaryOpLayer a3(BinaryOpLayer::Operation::kAdd);
  EWLayer a4("relu");

  graph.setInput(a2, input);
  graph.makeConnection(a2, a2_1);
  graph.makeConnection(a2, a2_2);
  graph.makeConnection(a2_1, a2_1_1);
  graph.makeConnection(a2_1_1, a2_1_1_1);
  graph.makeConnection(a2_1_1_1, a2_1_1_2);
  graph.makeConnection(a2_1_1_2, a2_1_2);
  graph.makeConnection(a2_1_1, a2_1_2);
  graph.makeConnection(a2_1_2, a2_1_3);
  graph.makeConnection(a2_1_3, a3);
  graph.makeConnection(a2_2, a3);
  graph.makeConnection(a3, a4);
  graph.setOutput(a4, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(4, 12);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_resnet2) {
  Graph graph(151);
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  SplitLayer a2(1, 2);

  EWLayer a2_1("relu");
  EWLayer a2_1_1("relu");

  EWLayer a2_1_1_1("relu");
  EWLayer a2_1_1_2("relu");

  BinaryOpLayer a2_1_2(BinaryOpLayer::Operation::kMul);
  EWLayer a2_1_3("relu");
  BinaryOpLayer a3(BinaryOpLayer::Operation::kAdd);
  EWLayer a4("relu");

  graph.setInput(a2, input);
  graph.makeConnection(a2, a2_1);
  graph.makeConnection(a2_1, a2_1_1);
  graph.makeConnection(a2_1_1, a2_1_1_1);
  graph.makeConnection(a2_1_1_1, a2_1_1_2);
  graph.makeConnection(a2_1_1_2, a2_1_2);
  graph.makeConnection(a2_1_1, a2_1_2);
  graph.makeConnection(a2_1_2, a2_1_3);
  graph.makeConnection(a2_1_3, a3);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a3, a4);
  graph.setOutput(a4, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(4, 12);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_google1) {
  Graph graph(151);
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  EWLayer a2("relu");

  EWLayer a2_1("relu");
  EWLayer a2_2("relu");
  EWLayer a2_3("relu");
  EWLayer a2_4("relu");

  EWLayer a2_2_1("linear", 2.0F, 3.0F);
  EWLayer a2_3_1("linear", 2.0F, 3.0F);

  ConcatLayer a3(0);

  graph.setInput(a2, input);
  graph.makeConnection(a2, a2_1);
  graph.makeConnection(a2, a2_2);
  graph.makeConnection(a2, a2_3);
  graph.makeConnection(a2, a2_4);
  graph.makeConnection(a2_2, a2_2_1);
  graph.makeConnection(a2_3, a2_3_1);
  graph.makeConnection(a2_4, a3);
  graph.makeConnection(a2_3_1, a3);
  graph.makeConnection(a2_2_1, a3);
  graph.makeConnection(a2_1, a3);
  graph.setOutput(a3, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(32, 3);
  for (int c = 8; c < 24; c++) {
    res[c] = 9;
  }
  ASSERT_EQ(tmp, res);
}

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
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a4(1, 0, 1, kernel);
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
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
  std::vector<float> kernelvec;
  kernelvec.reserve(3 * 3 * 3 * 3);
  for (int i = 0; i < 81; ++i) {
    kernelvec.push_back(1);
  }
  Shape sh2({3, 3, 3, 3});
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
  ASSERT_GT(tmp.size(), 0);
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_GE(tmp[i], 0);
  }
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
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a3(1, 0, 1, kernel);

  // EWLayer a4("linear", 2.0F, 3.0F);
  // a2.ewops.layers.push_back(&a4);
  // a2.ewops.countlayers++;
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
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);
  ConvolutionalLayer a3(1, 0, 1, kernel);

  EWLayer a4("linear", 2.0F, 3.0F);
  a2.postops.layers.push_back(&a4);
  a2.postops.count++;

  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.setOutput(a3, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {189, 189, 189};
  ASSERT_EQ(tmp, res);
}

FLAKY_TEST(bfs, check_struct_graph_split) {
  std::vector<std::vector<std::pair<int, int>>> split = {
      {{12, 0}, {13, 0}, {14, 0}}};
  Graph graph(151, split);
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer a1(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 1, kernel);

  ConvolutionalLayer a3_1(1, 0, 1, kernel);
  EWLayer a3_1_1("relu");
  ConvolutionalLayer a3_2(1, 0, 1, kernel);
  EWLayer a3_2_1("relu");

  ConcatLayer a4(0);
  EWLayer a5("relu");

  EWLayer a6_1("relu");
  EWLayer a6_2("relu");

  ConcatLayer a7(0);
  // EWLayer a8("relu");
  SplitLayer a8(1, 3);

  EWLayer a9_1("relu");
  EWLayer a9_2("relu");
  EWLayer a9_3("relu");

  ConcatLayer a10(0);
  EWLayer a11_1("relu");

  ConcatLayer a12(0);

  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3_1);
  graph.makeConnection(a2, a3_2);
  graph.makeConnection(a3_1, a3_1_1);
  graph.makeConnection(a3_1_1, a4);
  graph.makeConnection(a3_2, a3_2_1);
  graph.makeConnection(a3_2_1, a4);
  graph.makeConnection(a4, a5);
  graph.makeConnection(a5, a7);
  graph.makeConnection(a5, a6_1);
  graph.makeConnection(a5, a6_2);
  graph.makeConnection(a6_1, a7);
  graph.makeConnection(a6_2, a7);
  graph.makeConnection(a7, a8);
  graph.makeConnection(a8, a9_1);
  graph.makeConnection(a8, a9_2);
  graph.makeConnection(a8, a9_3);
  graph.makeConnection(a9_1, a10);
  graph.makeConnection(a9_2, a10);
  graph.makeConnection(a9_3, a10);
  graph.makeConnection(a10, a11_1);
  graph.makeConnection(a11_1, a12);
  graph.makeConnection(a10, a12);
  graph.setOutput(a12, output);
  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(36, 81);
  ASSERT_EQ(tmp, res);
}
FLAKY_END_TEST
