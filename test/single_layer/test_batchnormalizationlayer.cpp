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