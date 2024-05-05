#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "layers/OutputLayer.hpp"

using namespace itlab_2023;

void fill_from_file(const std::string& path_from, std::vector<std::string>& to,
                    size_t limit = 0) {
  to.clear();
  std::ifstream f;
  std::string buf;
  f.open(path_from, std::ios::in);
  if (f.fail()) {
    throw std::runtime_error("No such file");
  }
  while (!f.eof()) {
    std::getline(f, buf);
    to.emplace_back(buf);
    if (limit > 0 && to.size() >= limit) {
      break;
    }
  }
  f.close();
}

class OutputTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<float>, size_t, std::vector<float> > > {};
// 1) input; 2) k for top_k; 3) expected_output.

TEST_P(OutputTestsParameterized, output_layer_works_correctly) {
  auto data = GetParam();
  Tensor input = make_tensor(std::get<0>(data));
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels, input.get_shape().count());
  size_t k = std::get<1>(data);
  OutputLayer layer(labels);
  auto top_k = layer.top_k(input, k);
  std::vector<float> true_output = std::get<2>(data);
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*top_k.second.as<float>())[i], true_output[i], 1e-5);
  }
}

INSTANTIATE_TEST_SUITE_P(
    output_layer_tests, OutputTestsParameterized,
    ::testing::Values(
        std::make_tuple(std::vector<float>({2.0F, 3.9F, 0.1F, 2.3F}), 3,
                        std::vector<float>({3.9F, 2.3F, 2.0F})),
        std::make_tuple(std::vector<float>({1.0F, -1.0F, 2.0F, -2.0F}), 4,
                        std::vector<float>({2.0F, 1.0F, -1.0F, -2.0F}))));

TEST(OutputLayer, can_get_topk_with_vector) {
  const int k = 50;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<double> input;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(static_cast<double>(std::rand()) / RAND_MAX);
  }
  ASSERT_NO_THROW(auto topk1 = top_k_vec(input, labels, k));
}

TEST(OutputLayer, can_get_topk_with_layer_float) {
  const int k = 50;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<float> input;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(
        static_cast<float>(static_cast<double>(std::rand()) / RAND_MAX));
  }
  Tensor input_tensor = make_tensor(input);
  OutputLayer layer(labels);
  ASSERT_NO_THROW(auto topk1 = layer.top_k(input_tensor, k));
}

TEST(OutputLayer, can_get_topk_with_layer_int) {
  const int k = 50;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<int> input;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(std::rand());
  }
  Tensor input_tensor = make_tensor(input);
  OutputLayer layer(labels);
  ASSERT_NO_THROW(auto topk1 = layer.top_k(input_tensor, k));
}

TEST(OutputLayer, topk_throws_when_not_1d_input) {
  const int k = 50;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<int> input;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(std::rand());
  }
  Tensor input_tensor = make_tensor(input, {5, 200});
  OutputLayer layer(labels);
  ASSERT_ANY_THROW(auto topk1 = layer.top_k(input_tensor, k));
}

TEST(OutputLayer, topk_throws_when_incorrect_input_size) {
  const int k = 50;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<int> input;
  // get random nums
  for (size_t i = 0; i < 20; i++) {
    input.push_back(std::rand());
  }
  Tensor input_tensor = make_tensor(input);
  OutputLayer layer(labels);
  ASSERT_ANY_THROW(auto topk1 = layer.top_k(input_tensor, k));
}

TEST(OutputLayer, topk_throws_when_too_big_k) {
  const int k = 2500;
  std::vector<std::string> labels;
  fill_from_file(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt",
                 labels);
  std::vector<int> input;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(std::rand());
  }
  Tensor input_tensor = make_tensor(input);
  OutputLayer layer(labels);
  ASSERT_ANY_THROW(auto topk1 = layer.top_k(input_tensor, k));
}

TEST(OutputLayer, get_layer_name) {
  EXPECT_EQ(OutputLayer::get_name(), "Output layer");
}

TEST(OutputLayer, softmax_works) {
  std::vector<double> input = {1.0, 2.5, 4.0, 5.5};
  std::vector<double> converted_input = {0.008657, 0.038774, 0.173774,
                                         0.778800};
  std::vector<double> output = softmax<double>(input);
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(OutputLayer, softmax_throws_when_empty_input) {
  std::vector<double> input;
  ASSERT_ANY_THROW(softmax<double>(input));
}
