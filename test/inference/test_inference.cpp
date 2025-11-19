#include <memory>
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
  Graph graph;
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_1 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_1_1 = std::make_unique<EWLayer>("relu");
  auto a3_2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_2_1 = std::make_unique<EWLayer>("relu");
  auto a4 = std::make_unique<ConcatLayer>(0);
  auto a5 = std::make_unique<EWLayer>("relu");
  auto a6_1 = std::make_unique<EWLayer>("relu");
  auto a6_2 = std::make_unique<EWLayer>("relu");
  auto a7 = std::make_unique<ConcatLayer>(0);
  auto a8 = std::make_unique<SplitLayer>(1, 3);
  auto a9_1 = std::make_unique<EWLayer>("relu");
  auto a9_2 = std::make_unique<EWLayer>("relu");
  auto a9_3 = std::make_unique<EWLayer>("relu");
  auto a10 = std::make_unique<ConcatLayer>(0);
  auto a11_1 = std::make_unique<EWLayer>("relu");
  auto a12 = std::make_unique<ConcatLayer>(0);

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a3_1_ptr = a3_1.get();
  Layer* a3_1_1_ptr = a3_1_1.get();
  Layer* a3_2_ptr = a3_2.get();
  Layer* a3_2_1_ptr = a3_2_1.get();
  Layer* a4_ptr = a4.get();
  Layer* a5_ptr = a5.get();
  Layer* a6_1_ptr = a6_1.get();
  Layer* a6_2_ptr = a6_2.get();
  Layer* a7_ptr = a7.get();
  Layer* a8_ptr = a8.get();
  Layer* a9_1_ptr = a9_1.get();
  Layer* a9_2_ptr = a9_2.get();
  Layer* a9_3_ptr = a9_3.get();
  Layer* a10_ptr = a10.get();
  Layer* a11_1_ptr = a11_1.get();
  Layer* a12_ptr = a12.get();

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a3_1_ptr);
  graph.makeConnection(a2_ptr, a3_2_ptr);
  graph.makeConnection(a3_1_ptr, a3_1_1_ptr);
  graph.makeConnection(a3_1_1_ptr, a4_ptr);
  graph.makeConnection(a3_2_ptr, a3_2_1_ptr);
  graph.makeConnection(a3_2_1_ptr, a4_ptr);
  graph.makeConnection(a4_ptr, a5_ptr);
  graph.makeConnection(a5_ptr, a7_ptr);
  graph.makeConnection(a5_ptr, a6_1_ptr);
  graph.makeConnection(a5_ptr, a6_2_ptr);
  graph.makeConnection(a6_1_ptr, a7_ptr);
  graph.makeConnection(a6_2_ptr, a7_ptr);
  graph.makeConnection(a7_ptr, a8_ptr);
  graph.makeConnection(a8_ptr, a9_1_ptr);
  graph.makeConnection(a8_ptr, a9_2_ptr);
  graph.makeConnection(a8_ptr, a9_3_ptr);
  graph.makeConnection(a9_1_ptr, a10_ptr);
  graph.makeConnection(a9_2_ptr, a10_ptr);
  graph.makeConnection(a9_3_ptr, a10_ptr);
  graph.makeConnection(a10_ptr, a11_1_ptr);
  graph.makeConnection(a11_1_ptr, a12_ptr);
  graph.makeConnection(a10_ptr, a12_ptr);
  graph.setOutput(a12_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(36, 81);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_not_used_yolo) {
  Graph graph;
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

  auto a2 = std::make_unique<SplitLayer>(1, 4);
  auto a3_1 = std::make_unique<EWLayer>("relu");
  auto a3_1_1 = std::make_unique<EWLayer>("relu");
  auto a3_2 = std::make_unique<ConcatLayer>(0);
  auto a3_2_1 = std::make_unique<EWLayer>("relu");
  auto a3_3 = std::make_unique<EWLayer>("relu");
  auto a3_3_1 = std::make_unique<ConcatLayer>(0);
  auto a3_3_2 = std::make_unique<EWLayer>("relu");
  auto a3_3_3 = std::make_unique<EWLayer>("relu");
  auto a3_3_4 = std::make_unique<EWLayer>("relu");
  auto a4 = std::make_unique<ConcatLayer>(0);

  Layer* a2_ptr = a2.get();
  Layer* a3_1_ptr = a3_1.get();
  Layer* a3_1_1_ptr = a3_1_1.get();
  Layer* a3_2_ptr = a3_2.get();
  Layer* a3_2_1_ptr = a3_2_1.get();
  Layer* a3_3_ptr = a3_3.get();
  Layer* a3_3_1_ptr = a3_3_1.get();
  Layer* a3_3_2_ptr = a3_3_2.get();
  Layer* a3_3_3_ptr = a3_3_3.get();
  Layer* a3_3_4_ptr = a3_3_4.get();
  Layer* a4_ptr = a4.get();

  graph.setInput(a2_ptr, input);
  graph.makeConnection(a2_ptr, a3_1_ptr);
  graph.makeConnection(a2_ptr, a3_2_ptr);
  graph.makeConnection(a2_ptr, a3_3_ptr);
  graph.makeConnection(a3_1_ptr, a3_1_1_ptr);
  graph.makeConnection(a3_1_1_ptr, a4_ptr);
  graph.makeConnection(a3_2_ptr, a3_2_1_ptr);
  graph.makeConnection(a3_2_1_ptr, a4_ptr);
  graph.makeConnection(a3_3_ptr, a3_3_1_ptr);
  graph.makeConnection(a2_ptr, a3_3_1_ptr);
  graph.makeConnection(a3_3_1_ptr, a3_3_2_ptr);
  graph.makeConnection(a3_3_2_ptr, a3_3_3_ptr);
  graph.makeConnection(a3_3_3_ptr, a3_3_4_ptr);
  graph.makeConnection(a3_3_4_ptr, a3_2_ptr);
  graph.setOutput(a4_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(16, 3);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_resnet1) {
  Graph graph;
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a2 = std::make_unique<SplitLayer>(1, 2);
  auto a2_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1_2 = std::make_unique<EWLayer>("relu");
  auto a2_1_2 = std::make_unique<BinaryOpLayer>(BinaryOpLayer::Operation::kMul);
  auto a2_1_3 = std::make_unique<EWLayer>("relu");
  auto a2_2 = std::make_unique<EWLayer>("relu");
  auto a3 = std::make_unique<BinaryOpLayer>(BinaryOpLayer::Operation::kAdd);
  auto a4 = std::make_unique<EWLayer>("relu");

  Layer* a2_ptr = a2.get();
  Layer* a2_1_ptr = a2_1.get();
  Layer* a2_1_1_ptr = a2_1_1.get();
  Layer* a2_1_1_1_ptr = a2_1_1_1.get();
  Layer* a2_1_1_2_ptr = a2_1_1_2.get();
  Layer* a2_1_2_ptr = a2_1_2.get();
  Layer* a2_1_3_ptr = a2_1_3.get();
  Layer* a2_2_ptr = a2_2.get();
  Layer* a3_ptr = a3.get();
  Layer* a4_ptr = a4.get();

  graph.setInput(a2_ptr, input);
  graph.makeConnection(a2_ptr, a2_1_ptr);
  graph.makeConnection(a2_ptr, a2_2_ptr);
  graph.makeConnection(a2_1_ptr, a2_1_1_ptr);
  graph.makeConnection(a2_1_1_ptr, a2_1_1_1_ptr);
  graph.makeConnection(a2_1_1_1_ptr, a2_1_1_2_ptr);
  graph.makeConnection(a2_1_1_2_ptr, a2_1_2_ptr);
  graph.makeConnection(a2_1_1_ptr, a2_1_2_ptr);
  graph.makeConnection(a2_1_2_ptr, a2_1_3_ptr);
  graph.makeConnection(a2_1_3_ptr, a3_ptr);
  graph.makeConnection(a2_2_ptr, a3_ptr);
  graph.makeConnection(a3_ptr, a4_ptr);
  graph.setOutput(a4_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(4, 12);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_resnet2) {
  Graph graph;
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a2 = std::make_unique<SplitLayer>(1, 2);
  auto a2_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1_1 = std::make_unique<EWLayer>("relu");
  auto a2_1_1_2 = std::make_unique<EWLayer>("relu");
  auto a2_1_2 = std::make_unique<BinaryOpLayer>(BinaryOpLayer::Operation::kMul);
  auto a2_1_3 = std::make_unique<EWLayer>("relu");
  auto a3 = std::make_unique<BinaryOpLayer>(BinaryOpLayer::Operation::kAdd);
  auto a4 = std::make_unique<EWLayer>("relu");

  Layer* a2_ptr = a2.get();
  Layer* a2_1_ptr = a2_1.get();
  Layer* a2_1_1_ptr = a2_1_1.get();
  Layer* a2_1_1_1_ptr = a2_1_1_1.get();
  Layer* a2_1_1_2_ptr = a2_1_1_2.get();
  Layer* a2_1_2_ptr = a2_1_2.get();
  Layer* a2_1_3_ptr = a2_1_3.get();
  Layer* a3_ptr = a3.get();
  Layer* a4_ptr = a4.get();

  graph.setInput(a2_ptr, input);
  graph.makeConnection(a2_ptr, a2_1_ptr);
  graph.makeConnection(a2_1_ptr, a2_1_1_ptr);
  graph.makeConnection(a2_1_1_ptr, a2_1_1_1_ptr);
  graph.makeConnection(a2_1_1_1_ptr, a2_1_1_2_ptr);
  graph.makeConnection(a2_1_1_2_ptr, a2_1_2_ptr);
  graph.makeConnection(a2_1_1_ptr, a2_1_2_ptr);
  graph.makeConnection(a2_1_2_ptr, a2_1_3_ptr);
  graph.makeConnection(a2_1_3_ptr, a3_ptr);
  graph.makeConnection(a2_ptr, a3_ptr);
  graph.makeConnection(a3_ptr, a4_ptr);
  graph.setOutput(a4_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(4, 12);
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_graph_google1) {
  Graph graph;
  Shape sh1({1, 2, 2, 2});
  std::vector<int> vec;
  vec.reserve(8);
  for (int i = 0; i < 8; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a2 = std::make_unique<EWLayer>("relu");
  auto a2_1 = std::make_unique<EWLayer>("relu");
  auto a2_2 = std::make_unique<EWLayer>("relu");
  auto a2_3 = std::make_unique<EWLayer>("relu");
  auto a2_4 = std::make_unique<EWLayer>("relu");
  auto a2_2_1 = std::make_unique<EWLayer>("linear", 2.0F, 3.0F);
  auto a2_3_1 = std::make_unique<EWLayer>("linear", 2.0F, 3.0F);
  auto a3 = std::make_unique<ConcatLayer>(0);

  Layer* a2_ptr = a2.get();
  Layer* a2_1_ptr = a2_1.get();
  Layer* a2_2_ptr = a2_2.get();
  Layer* a2_3_ptr = a2_3.get();
  Layer* a2_4_ptr = a2_4.get();
  Layer* a2_2_1_ptr = a2_2_1.get();
  Layer* a2_3_1_ptr = a2_3_1.get();
  Layer* a3_ptr = a3.get();

  graph.setInput(a2_ptr, input);
  graph.makeConnection(a2_ptr, a2_1_ptr);
  graph.makeConnection(a2_ptr, a2_2_ptr);
  graph.makeConnection(a2_ptr, a2_3_ptr);
  graph.makeConnection(a2_ptr, a2_4_ptr);
  graph.makeConnection(a2_2_ptr, a2_2_1_ptr);
  graph.makeConnection(a2_3_ptr, a2_3_1_ptr);
  graph.makeConnection(a2_4_ptr, a3_ptr);
  graph.makeConnection(a2_3_1_ptr, a3_ptr);
  graph.makeConnection(a2_2_1_ptr, a3_ptr);
  graph.makeConnection(a2_1_ptr, a3_ptr);
  graph.setOutput(a3_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(32, 3);
  for (int c = 8; c < 24; c++) {
    res[c] = 9;
  }
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_result_vec) {
  Graph graph;
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  auto a3 = std::make_unique<InputLayer>(kNhwc, kNhwc, 1, 1);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a4 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a4_ptr = a4.get();

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a4_ptr);
  graph.setOutput(a4_ptr, output);

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
  Graph graph;
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  std::vector<float> kernelvec;
  kernelvec.reserve(3 * 3 * 3 * 3);
  for (int i = 0; i < 81; ++i) {
    kernelvec.push_back(1);
  }
  Shape sh2({3, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  Shape poolshape = {2, 2};
  auto a3 = std::make_unique<EWLayer>("linear", 2.0F, 3.0F);
  auto a4 = std::make_unique<PoolingLayer>(poolshape, "average");
  auto a6 = std::make_unique<FCLayer>();
  auto a5 = std::make_unique<OutputLayer>();

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a3_ptr = a3.get();
  Layer* a4_ptr = a4.get();
  Layer* a5_ptr = a5.get();

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a3_ptr);
  graph.makeConnection(a3_ptr, a4_ptr);
  graph.makeConnection(a4_ptr, a5_ptr);
  graph.setOutput(a5_ptr, output);

  graph.inference();

  std::vector<float> tmp = *output.as<float>();
  ASSERT_GT(tmp.size(), 0);
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_GE(tmp[i], 0);
  }
}

TEST(bfs, check_struct_layer) {
  Graph graph;
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a3_ptr = a3.get();

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a3_ptr);
  graph.setOutput(a3_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {81, 81, 81};
  ASSERT_EQ(tmp, res);
}

TEST(bfs, check_struct_layer_added) {
  Graph graph;
  Shape sh1({1, 5, 5, 3});
  std::vector<int> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a4 = std::make_unique<EWLayer>("linear", 2.0F, 3.0F);

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a3_ptr = a3.get();
  Layer* a4_ptr = a4.get();

  a2->postops.layers.push_back(a4_ptr);
  a2->postops.count++;

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a3_ptr);
  graph.setOutput(a3_ptr, output);

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

  auto a1 = std::make_unique<InputLayer>(kNhwc, kNchw, 1, 2);
  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  auto a2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_1 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_1_1 = std::make_unique<EWLayer>("relu");
  auto a3_2 = std::make_unique<ConvolutionalLayer>(1, 0, 1, kernel);
  auto a3_2_1 = std::make_unique<EWLayer>("relu");
  auto a4 = std::make_unique<ConcatLayer>(0);
  auto a5 = std::make_unique<EWLayer>("relu");
  auto a6_1 = std::make_unique<EWLayer>("relu");
  auto a6_2 = std::make_unique<EWLayer>("relu");
  auto a7 = std::make_unique<ConcatLayer>(0);
  auto a8 = std::make_unique<SplitLayer>(1, 3);
  auto a9_1 = std::make_unique<EWLayer>("relu");
  auto a9_2 = std::make_unique<EWLayer>("relu");
  auto a9_3 = std::make_unique<EWLayer>("relu");
  auto a10 = std::make_unique<ConcatLayer>(0);
  auto a11_1 = std::make_unique<EWLayer>("relu");
  auto a12 = std::make_unique<ConcatLayer>(0);

  Layer* a1_ptr = a1.get();
  Layer* a2_ptr = a2.get();
  Layer* a3_1_ptr = a3_1.get();
  Layer* a3_1_1_ptr = a3_1_1.get();
  Layer* a3_2_ptr = a3_2.get();
  Layer* a3_2_1_ptr = a3_2_1.get();
  Layer* a4_ptr = a4.get();
  Layer* a5_ptr = a5.get();
  Layer* a6_1_ptr = a6_1.get();
  Layer* a6_2_ptr = a6_2.get();
  Layer* a7_ptr = a7.get();
  Layer* a8_ptr = a8.get();
  Layer* a9_1_ptr = a9_1.get();
  Layer* a9_2_ptr = a9_2.get();
  Layer* a9_3_ptr = a9_3.get();
  Layer* a10_ptr = a10.get();
  Layer* a11_1_ptr = a11_1.get();
  Layer* a12_ptr = a12.get();

  graph.setInput(a1_ptr, input);
  graph.makeConnection(a1_ptr, a2_ptr);
  graph.makeConnection(a2_ptr, a3_1_ptr);
  graph.makeConnection(a2_ptr, a3_2_ptr);
  graph.makeConnection(a3_1_ptr, a3_1_1_ptr);
  graph.makeConnection(a3_1_1_ptr, a4_ptr);
  graph.makeConnection(a3_2_ptr, a3_2_1_ptr);
  graph.makeConnection(a3_2_1_ptr, a4_ptr);
  graph.makeConnection(a4_ptr, a5_ptr);
  graph.makeConnection(a5_ptr, a7_ptr);
  graph.makeConnection(a5_ptr, a6_1_ptr);
  graph.makeConnection(a5_ptr, a6_2_ptr);
  graph.makeConnection(a6_1_ptr, a7_ptr);
  graph.makeConnection(a6_2_ptr, a7_ptr);
  graph.makeConnection(a7_ptr, a8_ptr);
  graph.makeConnection(a8_ptr, a9_1_ptr);
  graph.makeConnection(a8_ptr, a9_2_ptr);
  graph.makeConnection(a8_ptr, a9_3_ptr);
  graph.makeConnection(a9_1_ptr, a10_ptr);
  graph.makeConnection(a9_2_ptr, a10_ptr);
  graph.makeConnection(a9_3_ptr, a10_ptr);
  graph.makeConnection(a10_ptr, a11_1_ptr);
  graph.makeConnection(a11_1_ptr, a12_ptr);
  graph.makeConnection(a10_ptr, a12_ptr);
  graph.setOutput(a12_ptr, output);

  graph.inference();
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res(36, 81);
  ASSERT_EQ(tmp, res);
}
FLAKY_END_TEST