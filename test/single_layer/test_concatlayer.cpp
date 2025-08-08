#include <vector>

#include "gtest/gtest.h"
#include "layers/ConCatLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(ConcatLayerTests, ConcatSameShapeFloatAxis0) {
  ConcatLayer layer;
  Tensor input1 = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor input2 = make_tensor<float>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({4, 2}));

  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1}), 4.0f);

  EXPECT_FLOAT_EQ(output.get<float>({2, 0}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({2, 1}), 6.0f);
  EXPECT_FLOAT_EQ(output.get<float>({3, 0}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({3, 1}), 8.0f);
}

TEST(ConcatLayerTests, ConcatSameShapeIntAxis1) {
  ConcatLayer layer(1);
  Tensor input1 = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  Tensor input2 = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 4}));

  EXPECT_EQ(output.get<int>({0, 0}), 1);
  EXPECT_EQ(output.get<int>({0, 1}), 2);
  EXPECT_EQ(output.get<int>({0, 2}), 1);
  EXPECT_EQ(output.get<int>({0, 3}), 2);

  EXPECT_EQ(output.get<int>({1, 0}), 3);
  EXPECT_EQ(output.get<int>({1, 1}), 4);
  EXPECT_EQ(output.get<int>({1, 2}), 3);
  EXPECT_EQ(output.get<int>({1, 3}), 4);
}

TEST(ConcatLayerTests, Concat3DTensorsAxis2) {
  ConcatLayer layer(2);
  Tensor input1 = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor input2 =
      make_tensor<float>({9, 10, 11, 12, 13, 14, 15, 16}, {2, 2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 2, 4}));

  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1}), 4.0f);

  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 2}), 9.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 3}), 10.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 2}), 11.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 3}), 12.0f);

  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 0}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 1}), 8.0f);

  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 2}), 13.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 0, 3}), 14.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 2}), 15.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1, 3}), 16.0f);
}

TEST(ConcatLayerTests, NegativeAxis) {
  ConcatLayer layer(-1);
  Tensor input1 = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor input2 = make_tensor<float>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 4}));

  EXPECT_FLOAT_EQ(output.get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 2}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 3}), 6.0f);

  EXPECT_FLOAT_EQ(output.get<float>({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 1}), 4.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 2}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({1, 3}), 8.0f);
}

TEST(ConcatLayerTests, ConcatResNetStyle) {
  ConcatLayer layer(1);
  Tensor input1 = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 2, 2});
  Tensor input2 =
      make_tensor<float>({9, 10, 11, 12, 13, 14, 15, 16}, {1, 2, 2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({1, 4, 2, 2}));

  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 0, 1, 1}), 4.0f);

  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 0}), 7.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 1, 1, 1}), 8.0f);

  EXPECT_FLOAT_EQ(output.get<float>({0, 2, 0, 0}), 9.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 2, 0, 1}), 10.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 2, 1, 0}), 11.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 2, 1, 1}), 12.0f);

  EXPECT_FLOAT_EQ(output.get<float>({0, 3, 0, 0}), 13.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 3, 0, 1}), 14.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 3, 1, 0}), 15.0f);
  EXPECT_FLOAT_EQ(output.get<float>({0, 3, 1, 1}), 16.0f);
}