#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

using namespace itlab_2023;

class EWTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, EWLayerImpl<double>,
                     std::vector<double>, std::function<double(double)> > > {};
// 1) input; 2) constructed ewlayerimpl; 3) expected_output; 4) lambda_expr.

TEST_P(EWTestsParameterized, element_wise_works_correctly) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  EWLayerImpl<double> a = std::get<1>(data);
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = std::get<2>(data);
  auto func = std::get<3>(data);
  if (func != nullptr) {
    true_output = std::vector<double>(input.size());
    std::transform(input.begin(), input.end(), true_output.begin(), func);
  }
  EXPECT_EQ(output, true_output);
}

std::vector<double> basic_data1 = {2.0, 3.9, 0.1, 2.3};
std::vector<double> basic_data2 = {1.0, -1.0, 2.0, -2.0};

INSTANTIATE_TEST_SUITE_P(
    element_wise_tests, EWTestsParameterized,
    ::testing::Values(
        std::make_tuple(basic_data1, EWLayerImpl<double>({2, 2}, "minus"),
                        std::vector<double>({-2.0, -3.9, -0.1, -2.3}),
                        std::function<double(double)>()),
        std::make_tuple(basic_data1, EWLayerImpl<double>({2, 2}, "sin"),
                        std::vector<double>(),
                        std::function<double(double)>([](double arg) -> double {
                          return std::sin(arg);
                        })),
        std::make_tuple(basic_data2, EWLayerImpl<double>({2, 2}, "relu"),
                        std::vector<double>({1.0, 0.0, 2.0, 0.0}),
                        std::function<double(double)>()),
        std::make_tuple(basic_data2, EWLayerImpl<double>({2, 2}, "tanh"),
                        std::vector<double>(),
                        std::function<double(double)>([](double arg) -> double {
                          return std::tanh(arg);
                        })),
        std::make_tuple(basic_data2,
                        EWLayerImpl<double>({2, 2}, "linear", 2.0F, 1.0F),
                        std::vector<double>({3.0, -1.0, 5.0, -3.0}),
                        std::function<double(double)>())));

TEST(ewlayer, new_ewlayer_can_relu_float) {
  EWLayer layer("relu");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  layer.run(input, output);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*output.as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_relu_int) {
  EWLayer layer("relu");
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output;
  std::vector<int> converted_input = {1, 0, 2, 0};
  layer.run(input, output);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*output.as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, new_ewlayer_can_linear_float) {
  EWLayer layer("linear", 2.0F, 3.0F);
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output = make_tensor<int>({0});
  std::vector<int> converted_input = {5, 1, 7, -1};
  layer.run(input, output);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*output.as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, new_ewlayer_throws_with_invalid_function) {
  EWLayer layer("abra");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  ASSERT_ANY_THROW(layer.run(input, output));
}

TEST(ewlayer, get_layer_name) {
  EXPECT_EQ(EWLayer::get_name(), "Element-wise layer");
}
