#include <gtest/gtest.h>

#include "layers/ReduceSumLayer.hpp"
#include "layers/Tensor.hpp"

namespace itlab_2023 {

TEST(ReduceSumLayer, DefaultConstructor) {
  ASSERT_NO_THROW(ReduceSumLayer layer);
}

TEST(ReduceSumLayer, SumAllAxesKeepDims) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor output;

  layer.run(input, output);

  EXPECT_EQ(output.get_shape(), Shape({1, 1}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 10.0f);
}

TEST(ReduceSumLayer, SumAlongAxis0) {
  ReduceSumLayer layer(0);
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({1});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(output.get<float>({0}), 4.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1}), 6.0f);
}

TEST(ReduceSumLayer, SumAlongAxis1KeepDims) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({2});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({2, 1}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 7.0f);
}

TEST(ReduceSumLayer, InvalidAxisThrows) {
  ReduceSumLayer layer;
  Tensor input = make_tensor<float>({1.0f, 2.0f}, {2});
  Tensor axes = make_tensor<int>({3});

  Tensor output;
  ASSERT_THROW(layer.run(input, axes, output), std::runtime_error);
}

TEST(ReduceSumLayer, IntTensorSupport) {
  ReduceSumLayer layer(0);
  Tensor input = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  Tensor axes = make_tensor<int>({1});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({2}));
  EXPECT_EQ(output.get<int>({0}), 4);
  EXPECT_EQ(output.get<int>({1}), 6);
}

TEST(ReduceSumLayer, 3DTensorReduction) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({3});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({2, 2, 1}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 0}), 11.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 0}), 15.0f);
}

TEST(ReduceSumLayer, 3DReductionAxis2) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({2});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({2, 1, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 1}), 14.0f);
}

TEST(ReduceSumLayer, 3DReductionAxis10) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {2, 2, 2, 2});

  Tensor axes = make_tensor<int>({1});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({1, 2, 2, 2}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 0}), 1 + 9);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 1}), 2 + 10);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 0}), 3 + 11);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 1}), 4 + 12);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 0}), 5 + 13);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 1}), 6 + 14);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 0}), 7 + 15);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 1}), 8 + 16);
}

TEST(ReduceSumLayer, 3DFullReduction) {
  ReduceSumLayer layer(1);
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});

  Tensor output;
  layer.run(input, output);

  EXPECT_EQ(output.get_shape(), Shape({1, 1, 1}));
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0}), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}

TEST(ReduceSumLayer, Resnet) {
  ReduceSumLayer layer(0);
  Tensor input = make_tensor<int>({1, 2, 64, 64, 64}, {5});
  Tensor axes = make_tensor<int>({1});
  Tensor output;

  layer.run(input, axes, output);

  EXPECT_EQ(output.get_shape(), Shape({1}));
  EXPECT_EQ(output.get<int>({0}), 195);
}

}  // namespace itlab_2023