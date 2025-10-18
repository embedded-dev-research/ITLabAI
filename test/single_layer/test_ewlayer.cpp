#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

using namespace it_lab_ai;

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
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], true_output[i], 1e-5);
  }
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
                        std::function<double(double)>()),
        std::make_tuple(std::vector<double>({0.0, 1.0, -1.0}),
                        EWLayerImpl<double>({3}, "sigmoid"),
                        std::vector<double>(),
                        std::function<double(double)>([](double x) {
                          return 1.0 / (1.0 + std::exp(-x));
                        })),
        std::make_tuple(std::vector<double>{-100.0, -50.0, 0.0, 50.0, 100.0},
                        EWLayerImpl<double>({5}, "sigmoid"),
                        std::vector<double>(),
                        std::function<double(double)>([](double x) {
                          if (x >= 0) {
                            double z = std::exp(-x);
                            return 1.0 / (1.0 + z);
                          } else {
                            double z = std::exp(x);
                            return z / (1.0 + z);
                          }
                        }))));

TEST(ewlayer, new_ewlayer_can_relu_float) {
  EWLayer layer("relu");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_mul_float) {
  EWLayer layer("linear", 2.0f, 0.0f);
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> converted_input = {2.0F, -2.0F, 4.0F, -10.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_sub_float) {
  EWLayer layer("linear", 1.0f, -1.0f);
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> converted_input = {0.0F, -2.0F, 1.0F, -6.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_relu_int) {
  EWLayer layer("relu");
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output;
  std::vector<int> converted_input = {1, 0, 2, 0};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*out[0].as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, new_ewlayer_can_linear_float) {
  EWLayer layer("linear", 2.0F, 3.0F);
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output = make_tensor<int>({0});
  std::vector<int> converted_input = {5, 1, 7, -1};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*out[0].as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, IncompatibleInput) {
  EWLayer layer("abra");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input, input};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(ewlayer, new_ewlayer_throws_with_invalid_function) {
  EWLayer layer("abra");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(ewlayer, new_ewlayer_can_sigmoid_float) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<float>({0.0F, -1.0F, 1.0F, 2.0F});
  Tensor output;
  std::vector<float> expected_output = {0.5F, 1.0F / (1.0F + std::exp(1.0F)),
                                        1.0F / (1.0F + std::exp(-1.0F)),
                                        1.0F / (1.0F + std::exp(-2.0F))};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], expected_output[i], 1e-5F);
  }
}

TEST(ewlayer, new_ewlayer_can_sigmoid_int) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<int>({0, -100, 100, 1, -1});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<int> expected = {1, 0, 1, 1, 0};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*out[0].as<int>())[i], expected[i]);
  }
}

TEST(ewlayer, new_ewlayer_can_sigmoid_float_extreme_values) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<float>({0.0F, -1.0F, 1.0F, 2.0F, -100.0F, 100.0F});
  Tensor output;

  auto stable_sigmoid = [](float x) {
    if (x >= 0) {
      float z = std::exp(-x);
      return 1.0F / (1.0F + z);
    } else {
      float z = std::exp(x);
      return z / (1.0F + z);
    }
  };

  std::vector<float> expected_output = {
      stable_sigmoid(0.0F), stable_sigmoid(-1.0F),   stable_sigmoid(1.0F),
      stable_sigmoid(2.0F), stable_sigmoid(-100.0F), stable_sigmoid(100.0F)};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  for (size_t i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], expected_output[i], 1e-5F);
  }
}
