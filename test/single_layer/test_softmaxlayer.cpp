#include <vector>

#include "gtest/gtest.h"
#include "layers/SoftmaxLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(SoftmaxLayerTest, BasicSoftmax1D) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Tensor input = make_tensor(data, {3});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({3}));

  float sum =
      out[0].get<float>({0}) + out[0].get<float>({1}) + out[0].get<float>({2});
  EXPECT_NEAR(sum, 1.0f, 1e-6);

  EXPECT_GT(out[0].get<float>({2}), out[0].get<float>({1}));
  EXPECT_GT(out[0].get<float>({1}), out[0].get<float>({0}));
}

TEST(SoftmaxLayerTest, Softmax2DAxis0) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(data, {2, 2});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2}));

  for (size_t col = 0; col < 2; ++col) {
    float sum = out[0].get<float>({0, col}) + out[0].get<float>({1, col});
    EXPECT_NEAR(sum, 1.0f, 1e-6);
  }
}

TEST(SoftmaxLayerTest, Softmax2DAxis1) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(data, {2, 2});
  Tensor output;
  SoftmaxLayer layer(1);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2}));

  for (size_t row = 0; row < 2; ++row) {
    float sum = out[0].get<float>({row, 0}) + out[0].get<float>({row, 1});
    EXPECT_NEAR(sum, 1.0f, 1e-6);
  }
}

TEST(SoftmaxLayerTest, Softmax3D) {
  std::vector<float> data(2 * 3 * 4, 1.0f);
  Tensor input = make_tensor(data, {2, 3, 4});
  Tensor output;
  SoftmaxLayer layer(1);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 3, 4}));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t k = 0; k < 4; ++k) {
      float sum = 0.0f;
      for (size_t j = 0; j < 3; ++j) {
        sum += out[0].get<float>({i, j, k});
      }
      EXPECT_NEAR(sum, 1.0f, 1e-6);
    }
  }
}

TEST(SoftmaxLayerTest, NegativeAxis) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(data, {2, 2});
  Tensor output;
  SoftmaxLayer layer(-1);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2}));

  for (size_t row = 0; row < 2; ++row) {
    float sum = out[0].get<float>({row, 0}) + out[0].get<float>({row, 1});
    EXPECT_NEAR(sum, 1.0f, 1e-6);
  }
}

TEST(SoftmaxLayerTest, IntTensorSoftmax) {
  std::vector<int> data = {1, 2, 3};
  Tensor input = make_tensor(data, {3});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({3}));
  ASSERT_EQ(out[0].get_type(), Type::kInt);

  EXPECT_GT(out[0].get<int>({2}), out[0].get<int>({1}));
  EXPECT_GT(out[0].get<int>({1}), out[0].get<int>({0}));
}

TEST(SoftmaxLayerTest, InvalidAxisError) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Tensor input = make_tensor(data, {3});
  Tensor output;
  SoftmaxLayer layer(5);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(SoftmaxLayerTest, MultipleInputsError) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Tensor input1 = make_tensor(data, {3});
  Tensor input2 = make_tensor(data, {3});
  Tensor output;
  SoftmaxLayer layer;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(SoftmaxLayerTest, LargeValuesStability) {
  std::vector<float> data = {1000.0f, 1001.0f, 1002.0f};
  Tensor input = make_tensor(data, {3});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  float sum =
      out[0].get<float>({0}) + out[0].get<float>({1}) + out[0].get<float>({2});
  EXPECT_NEAR(sum, 1.0f, 1e-6);
}

TEST(SoftmaxLayerTest, ExtremeNegativeAxis) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(data, {2, 2});
  Tensor output;
  SoftmaxLayer layer(-10);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(SoftmaxLayerTest, LargePositiveAxis) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input = make_tensor(data, {2, 2});
  Tensor output;

  SoftmaxLayer layer(5);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(SoftmaxLayerTest, AxisNormalizationVariants) {
  std::vector<float> data(2 * 3 * 4, 1.0f);
  Tensor input = make_tensor(data, {2, 3, 4});
  Tensor output;

  std::vector<int> axes = {-1, 2, -3, 0};

  for (int axis : axes) {
    SoftmaxLayer layer(axis);
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    if (axis == -3 || axis == 0) {
      EXPECT_NO_THROW(layer.run(in, out));

      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          float sum = 0.0f;
          for (size_t k = 0; k < 2; ++k) {
            sum += out[0].get<float>({k, i, j});
          }
          EXPECT_NEAR(sum, 1.0f, 1e-6);
        }
      }
    } else {
      EXPECT_NO_THROW(layer.run(in, out));
    }
  }
}

TEST(SoftmaxLayerTest, NumericalStabilityExtremeValues) {
  std::vector<float> large_values = {10000.0f, 10001.0f, 10002.0f};
  Tensor input_large = make_tensor(large_values, {3});
  Tensor output_large;
  SoftmaxLayer layer_large(0);

  std::vector<Tensor> in_large{input_large};
  std::vector<Tensor> out_large{output_large};

  EXPECT_NO_THROW(layer_large.run(in_large, out_large));

  float sum_large = out_large[0].get<float>({0}) +
                    out_large[0].get<float>({1}) + out_large[0].get<float>({2});
  EXPECT_NEAR(sum_large, 1.0f, 1e-6);

  for (size_t i = 0; i < 3; ++i) {
    float val = out_large[0].get<float>({i});
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);
  }
}

TEST(SoftmaxLayerTest, NumericalStabilityNegativeValues) {
  std::vector<float> negative_values = {-1000.0f, -1001.0f, -1002.0f};
  Tensor input_neg = make_tensor(negative_values, {3});
  Tensor output_neg;
  SoftmaxLayer layer_neg(0);

  std::vector<Tensor> in_neg{input_neg};
  std::vector<Tensor> out_neg{output_neg};

  EXPECT_NO_THROW(layer_neg.run(in_neg, out_neg));

  float sum_neg = out_neg[0].get<float>({0}) + out_neg[0].get<float>({1}) +
                  out_neg[0].get<float>({2});
  EXPECT_NEAR(sum_neg, 1.0f, 1e-6);
}

TEST(SoftmaxLayerTest, NumericalStabilityMixedValues) {
  std::vector<float> mixed_values = {-100.0f, 0.0f, 100.0f};
  Tensor input_mixed = make_tensor(mixed_values, {3});
  Tensor output_mixed;
  SoftmaxLayer layer_mixed(0);

  std::vector<Tensor> in_mixed{input_mixed};
  std::vector<Tensor> out_mixed{output_mixed};

  EXPECT_NO_THROW(layer_mixed.run(in_mixed, out_mixed));

  float sum_mixed = out_mixed[0].get<float>({0}) +
                    out_mixed[0].get<float>({1}) + out_mixed[0].get<float>({2});
  EXPECT_NEAR(sum_mixed, 1.0f, 1e-6);

  EXPECT_GT(out_mixed[0].get<float>({2}), out_mixed[0].get<float>({1}));
  EXPECT_GT(out_mixed[0].get<float>({1}), out_mixed[0].get<float>({0}));
}

TEST(SoftmaxLayerTest, VerifyMaxSubtraction) {
  std::vector<float> very_large = {1e10f, 1e10f + 1.0f, 1e10f + 2.0f};
  Tensor input = make_tensor(very_large, {3});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  for (size_t i = 0; i < 3; ++i) {
    float val = out[0].get<float>({i});
    EXPECT_FALSE(std::isnan(val));
    EXPECT_FALSE(std::isinf(val));
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);
  }
}

TEST(SoftmaxLayerTest, IntTensorExtremeValues) {
  std::vector<int> large_ints = {std::numeric_limits<int>::max() - 2,
                                 std::numeric_limits<int>::max() - 1,
                                 std::numeric_limits<int>::max()};
  Tensor input = make_tensor(large_ints, {3});
  Tensor output;
  SoftmaxLayer layer(0);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  for (size_t i = 0; i < 3; ++i) {
    int val = out[0].get<int>({i});
    EXPECT_GE(val, 0);
  }
}