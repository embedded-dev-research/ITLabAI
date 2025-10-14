#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/BatchNormalizationLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(BatchNormalizationLayerTest, EmptyInput) {
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({0.0f}, {1});
  Tensor var = make_tensor<float>({1.0f}, {1});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor input = make_tensor<float>({}, {0});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(BatchNormalizationLayerTest, WrongNumberOfInputs) {
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({0.0f}, {1});
  Tensor var = make_tensor<float>({1.0f}, {1});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor input1 = make_tensor<float>({1.0f}, {1});
  Tensor input2 = make_tensor<float>({2.0f}, {1});
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(BatchNormalizationLayerTest, ParameterShapeMismatch) {
  Tensor input = make_tensor<float>({1.0f, 2.0f}, {1, 2, 1, 1});

  Tensor scale = make_tensor<float>({1.0f, 1.0f, 1.0f}, {3});
  Tensor bias = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor mean = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor var = make_tensor<float>({1.0f, 1.0f}, {2});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(BatchNormalizationLayerTest, IdentityNormalization) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
  Tensor input = make_tensor<float>(input_data, {1, 2, 2, 2});

  Tensor scale = make_tensor<float>({1.0f, 1.0f}, {2});
  Tensor bias = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor mean = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor var = make_tensor<float>({1.0f, 1.0f}, {2});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 2, 2, 2}));

  for (size_t i = 0; i < input_data.size(); ++i) {
    EXPECT_NEAR(out[0].as<float>()->at(i), input_data[i], 1e-4);
  }
}

TEST(BatchNormalizationLayerTest, ScaleAndBias) {
  Tensor input = make_tensor<float>({1.0f, 1.0f, 1.0f, 1.0f}, {1, 2, 2, 1});

  Tensor scale = make_tensor<float>({2.0f, 2.0f}, {2});
  Tensor bias = make_tensor<float>({1.0f, 1.0f}, {2});
  Tensor mean = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor var = make_tensor<float>({1.0f, 1.0f}, {2});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 2, 2, 1}));

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(out[0].as<float>()->at(i), 3.0f, 1e-4);
  }
}

TEST(BatchNormalizationLayerTest, MeanAndVariance) {
  Tensor input = make_tensor<float>({4.0f, 5.0f, 6.0f, 5.0f}, {1, 2, 2, 1});

  Tensor scale = make_tensor<float>({1.0f, 1.0f}, {2});
  Tensor bias = make_tensor<float>({0.0f, 0.0f}, {2});
  Tensor mean = make_tensor<float>({5.0f, 5.0f}, {2});
  Tensor var = make_tensor<float>({1.0f, 1.0f}, {2});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 2, 2, 1}));

  EXPECT_NEAR(out[0].get<float>({0, 0, 0, 0}), -1.0f, 1e-5);
  EXPECT_NEAR(out[0].get<float>({0, 0, 1, 0}), 0.0f, 1e-5);
  EXPECT_NEAR(out[0].get<float>({0, 1, 0, 0}), 1.0f, 1e-5);
  EXPECT_NEAR(out[0].get<float>({0, 1, 1, 0}), 0.0f, 1e-5);
}

TEST(BatchNormalizationLayerTest, DifferentChannels) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f}, {1, 3, 1, 1});

  Tensor scale = make_tensor<float>({2.0f, 3.0f, 4.0f}, {3});
  Tensor bias = make_tensor<float>({1.0f, 2.0f, 3.0f}, {3});
  Tensor mean = make_tensor<float>({0.0f, 0.0f, 0.0f}, {3});
  Tensor var = make_tensor<float>({1.0f, 1.0f, 1.0f}, {3});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 3, 1, 1}));

  EXPECT_NEAR(out[0].get<float>({0, 0, 0, 0}), 1.0f * 2.0f + 1.0f, 1e-4);
  EXPECT_NEAR(out[0].get<float>({0, 1, 0, 0}), 2.0f * 3.0f + 2.0f, 1e-4);
  EXPECT_NEAR(out[0].get<float>({0, 2, 0, 0}), 3.0f * 4.0f + 3.0f, 1e-4);
}

TEST(BatchNormalizationLayerTest, EpsilonEffect) {
  Tensor input = make_tensor<float>({1.0f, 1.0001f}, {1, 1, 2, 1});
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({1.0f}, {1});
  Tensor var = make_tensor<float>({1e-12f}, {1});

  BatchNormalizationLayer layer(scale, bias, mean, var, 1e-6f);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 1, 2, 1}));

  EXPECT_FALSE(std::isnan(out[0].get<float>({0, 0, 0, 0})));
  EXPECT_FALSE(std::isinf(out[0].get<float>({0, 0, 0, 0})));
  EXPECT_FALSE(std::isnan(out[0].get<float>({0, 0, 1, 0})));
  EXPECT_FALSE(std::isinf(out[0].get<float>({0, 0, 1, 0})));
}

TEST(BatchNormalizationLayerTest, TrainingModeNotSupported) {
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({0.0f}, {1});
  Tensor var = make_tensor<float>({1.0f}, {1});

  BatchNormalizationLayer layer(scale, bias, mean, var, 1e-5f, 0.9f, true);
  Tensor input = make_tensor<float>({1.0f}, {1, 1, 1, 1});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(BatchNormalizationLayerTest, IntDataType) {
  Tensor input = make_tensor<int>({10, 20}, {1, 1, 2, 1});
  Tensor scale = make_tensor<int>({2}, {1});
  Tensor bias = make_tensor<int>({5}, {1});
  Tensor mean = make_tensor<int>({0}, {1});
  Tensor var = make_tensor<int>({1}, {1});

  BatchNormalizationLayer layer(scale, bias, mean, var);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 1, 2, 1}));

  EXPECT_EQ(out[0].get<int>({0, 0, 0, 0}), 10 * 2 + 5);
  EXPECT_EQ(out[0].get<int>({0, 0, 1, 0}), 20 * 2 + 5);
}

TEST(BatchNormalizationLayerTest, DifferentEpsilonValues) {
  Tensor input = make_tensor<float>({2.0f}, {1, 1, 1, 1});
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({1.0f}, {1});
  Tensor var = make_tensor<float>({1.0f}, {1});

  BatchNormalizationLayer layer1(scale, bias, mean, var, 0.1f);
  BatchNormalizationLayer layer2(scale, bias, mean, var, 1e-6f);

  Tensor output1, output2;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out1{output1};
  std::vector<Tensor> out2{output2};

  layer1.run(in, out1);
  layer2.run(in, out2);

  float result1 = out1[0].get<float>({0, 0, 0, 0});
  float result2 = out2[0].get<float>({0, 0, 0, 0});

  EXPECT_NE(result1, result2);
  EXPECT_GT(result2, result1);
}

TEST(BatchNormalizationLayerTest, ExactFormulaValidation) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(input_data, {1, 2, 2, 1});

  std::vector<float> scale = {2.0f, 0.5f};
  std::vector<float> bias = {1.0f, -1.0f};
  std::vector<float> mean = {2.0f, 3.0f};
  std::vector<float> var = {1.0f, 4.0f};
  float epsilon = 1e-5f;

  Tensor scale_tensor = make_tensor(scale, {2});
  Tensor bias_tensor = make_tensor(bias, {2});
  Tensor mean_tensor = make_tensor(mean, {2});
  Tensor var_tensor = make_tensor(var, {2});

  BatchNormalizationLayer layer(scale_tensor, bias_tensor, mean_tensor,
                                var_tensor, epsilon, 0.9f, false);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  float expected_ch0_0 =
      2.0f * (1.0f - 2.0f) / std::sqrt(1.0f + epsilon) + 1.0f;
  float expected_ch0_1 =
      2.0f * (2.0f - 2.0f) / std::sqrt(1.0f + epsilon) + 1.0f;

  float expected_ch1_0 =
      0.5f * (3.0f - 3.0f) / std::sqrt(4.0f + epsilon) - 1.0f;
  float expected_ch1_1 =
      0.5f * (4.0f - 3.0f) / std::sqrt(4.0f + epsilon) - 1.0f;

  EXPECT_NEAR(out[0].get<float>({0, 0, 0, 0}), expected_ch0_0, 1e-5f);
  EXPECT_NEAR(out[0].get<float>({0, 0, 1, 0}), expected_ch0_1, 1e-5f);
  EXPECT_NEAR(out[0].get<float>({0, 1, 0, 0}), expected_ch1_0, 1e-5f);
  EXPECT_NEAR(out[0].get<float>({0, 1, 1, 0}), expected_ch1_1, 1e-5f);
}

TEST(BatchNormalizationLayerTest, BroadcastingValidation) {
  std::vector<float> input_data(2 * 3 * 4 * 5, 2.0f);
  Tensor input = make_tensor(input_data, {2, 3, 4, 5});

  std::vector<float> scale = {1.0f, 2.0f, 3.0f};
  std::vector<float> bias = {0.1f, 0.2f, 0.3f};
  std::vector<float> mean = {1.0f, 1.5f, 2.0f};
  std::vector<float> var = {1.0f, 1.0f, 1.0f};

  Tensor scale_tensor = make_tensor(scale, {3});
  Tensor bias_tensor = make_tensor(bias, {3});
  Tensor mean_tensor = make_tensor(mean, {3});
  Tensor var_tensor = make_tensor(var, {3});

  BatchNormalizationLayer layer(scale_tensor, bias_tensor, mean_tensor,
                                var_tensor, 1e-5f, 0.9f, false);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  for (size_t b = 0; b < 2; ++b) {
    for (size_t c = 0; c < 3; ++c) {
      float expected =
          scale[c] * (2.0f - mean[c]) / std::sqrt(var[c] + 1e-5f) + bias[c];
      float first_val = out[0].get<float>({b, c, 0, 0});

      for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 5; ++w) {
          EXPECT_NEAR(out[0].get<float>({b, c, h, w}), first_val, 1e-5f);
          EXPECT_NEAR(out[0].get<float>({b, c, h, w}), expected, 1e-5f);
        }
      }
    }
  }
}

TEST(BatchNormalizationLayerTest, NumericalStabilityExtremeCases) {
  struct TestCase {
    float input;
    float var;
    const char* description;
  };

  std::vector<TestCase> test_cases = {
      {1e10f, 1e-10f, "very large input, very small variance"},
      {1e-10f, 1e10f, "very small input, very large variance"},
      {0.0f, 0.0f, "zero input and variance"},
      {-1e10f, 1e-10f, "very negative input, very small variance"}};

  for (const auto& tc : test_cases) {
    Tensor input = make_tensor<float>({tc.input}, {1, 1, 1, 1});
    Tensor scale = make_tensor<float>({1.0f}, {1});
    Tensor bias = make_tensor<float>({0.0f}, {1});
    Tensor mean = make_tensor<float>({0.0f}, {1});
    Tensor var = make_tensor<float>({tc.var}, {1});

    BatchNormalizationLayer layer(scale, bias, mean, var, 1e-5f, 0.9f, false);
    Tensor output;

    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out)) << "Failed for: " << tc.description;

    float result = out[0].get<float>({0, 0, 0, 0});
    EXPECT_FALSE(std::isnan(result)) << "NaN for: " << tc.description;
    EXPECT_FALSE(std::isinf(result)) << "Inf for: " << tc.description;
  }
}

TEST(BatchNormalizationLayerTest, DivisionByZeroProtection) {
  Tensor input = make_tensor<float>({5.0f}, {1, 1, 1, 1});
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({0.0f}, {1});
  Tensor var = make_tensor<float>({0.0f}, {1});

  BatchNormalizationLayer layer1(scale, bias, mean, var, 1e-10f, 0.9f, false);
  BatchNormalizationLayer layer2(scale, bias, mean, var, 1e-5f, 0.9f, false);

  Tensor output1, output2;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out1{output1}, out2{output2};

  EXPECT_NO_THROW(layer1.run(in, out1));
  EXPECT_NO_THROW(layer2.run(in, out2));

  float result1 = out1[0].get<float>({0, 0, 0, 0});
  float result2 = out2[0].get<float>({0, 0, 0, 0});

  EXPECT_NE(result1, result2);
  EXPECT_GT(std::abs(result1), std::abs(result2));
}