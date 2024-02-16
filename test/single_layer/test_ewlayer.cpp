#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

TEST(ewlayer, works_with_minus) {
  EWLayer<double> layer({2, 2}, "minus");
  std::vector<double> input = {2.0, 3.9, 0.1, 2.3};
  std::vector<double> converted_input = {-2.0, -3.9, -0.1, -2.3};
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, works_with_sin) {
  EWLayer<double> layer({2, 2}, "sin");
  std::vector<double> input = {2.0, 3.9, 0.1, 2.3};
  std::vector<double> converted_input(4);
  std::transform(input.begin(), input.end(), converted_input.begin(),
                 sin<double>);
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, relu_test) {
  EWLayer<double> layer({2, 2}, "relu");
  std::vector<double> input = {1.0, -1.0, 2.0, -2.0};
  std::vector<double> converted_input = {1.0, 0.0, 2.0, 0.0};
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, tanh_test) {
  EWLayer<double> layer({2, 2}, "tanh");
  std::vector<double> input = {1.0, -1.0, 2.0, -2.0};
  std::vector<double> converted_input(4);
  std::transform(input.begin(), input.end(), converted_input.begin(),
                 tanh<double>);
  std::vector<double> output = layer.run(input);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_NEAR(output[i], converted_input[i], 1e-5);
  }
}
