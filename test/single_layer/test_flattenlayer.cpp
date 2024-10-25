#include <vector>

#include "gtest/gtest.h"
#include "layers/FlattenLayer.hpp"

using namespace itlab_2023;

TEST(flattenlayer, new_flattenlayer_can_flatten_int) {
  FlattenLayer layer;
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  layer.run(input, output);
  EXPECT_EQ(output.get_shape().dims(), 1);
  EXPECT_EQ(output.get_shape()[0], 4);
}

TEST(flattenlayer, new_flattenlayer_can_flatten_float) {
  FlattenLayer layer;
  Shape sh({2, 2});
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F}, sh);
  Tensor output;
  layer.run(input, output);
  EXPECT_EQ(output.get_shape().dims(), 1);
  EXPECT_EQ(output.get_shape()[0], 4);
}

TEST(flattenlayer, get_layer_name) {
  EXPECT_EQ(FlattenLayer::get_name(), "Flatten layer");
}
