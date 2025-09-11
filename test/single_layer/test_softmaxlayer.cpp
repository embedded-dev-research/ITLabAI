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