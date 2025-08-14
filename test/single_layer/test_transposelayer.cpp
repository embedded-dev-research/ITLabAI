#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/Tensor.hpp"
#include "layers/TransposeLayer.hpp"

using namespace it_lab_ai;

TEST(TransposeLayerTest, EmptyTensor) {
  Tensor input = make_tensor<float>({}, {0});
  TransposeLayer layer;
  Tensor output;

  EXPECT_THROW(layer.run(input, output), std::runtime_error);
}

TEST(TransposeLayerTest, IdentityTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 1});
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1}), 4.0f);
}

TEST(TransposeLayerTest, VectorTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {4});
  TransposeLayer layer({0});
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({4}));
  EXPECT_FLOAT_EQ(output.get<float>({0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({2}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({3}), 4.0f);
}

TEST(TransposeLayerTest, InvalidPermutationSize) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0});
  Tensor output;

  EXPECT_THROW(layer.run(input, output), std::invalid_argument);
}

TEST(TransposeLayerTest, DuplicateAxes) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 0});
  Tensor output;

  EXPECT_THROW(layer.run(input, output), std::invalid_argument);
}

TEST(TransposeLayerTest, NegativeAxis) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, -1});
  Tensor output;

  EXPECT_THROW(layer.run(input, output), std::invalid_argument);
}

TEST(TransposeLayerTest, LargeAxis) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 2});
  Tensor output;

  EXPECT_THROW(layer.run(input, output), std::invalid_argument);
}

TEST(TransposeLayerTest, 4DTensorTranspose) {
  std::vector<float> data(16);
  std::iota(data.begin(), data.end(), 1.0f);
  Tensor input = make_tensor<float>(data, {2, 2, 2, 2});
  TransposeLayer layer({3, 1, 0, 2});
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 2, 2, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 1}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 0, 1}), 8.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 0}), 9.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 1, 0}), 10.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 0}), 13.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 1, 0}), 14.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 1}), 11.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 1, 1}), 12.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 1}), 15.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 1, 1}), 16.0f);
}

TEST(TransposeLayerTest, DefaultPermutation) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  TransposeLayer layer;
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({3, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({2, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({2, 1}), 6.0f);
}

TEST(TransposeLayerTest, MatrixTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({1, 0});
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1}), 4.0f);
}

TEST(TransposeLayerTest, 3DTensor) {
  std::vector<float> data(24);
  std::iota(data.begin(), data.end(), 1.0f);
  Tensor input = make_tensor<float>(data, {2, 3, 4});
  TransposeLayer layer({2, 0, 1});
  Tensor output;

  layer.run(input, output);

  ASSERT_EQ(output.get_shape(), Shape({4, 2, 3}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({3, 1, 2}), 24.0f);
}