#include <vector>

#include "gtest/gtest.h"
#include "layers/DropOutLayer.hpp"

using namespace itlab_2023;

TEST(DropOutLayer, dropoutlayer_int) {
  DropOutLayer layer(1);
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  layer.run(input, output);
  std::vector<int> vec = *output.as<int>();
  EXPECT_EQ(vec[0], 0);
  EXPECT_EQ(vec[1], 0);
  EXPECT_EQ(vec[2], 0);
  EXPECT_EQ(vec[3], 0);
}

TEST(DropOutLayer, dropoutlayer_float) {
  DropOutLayer layer(0);
  Shape sh({2, 2});
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F}, sh);
  Tensor output;
  layer.run(input, output);
  std::vector<float> vec = *output.as<float>();
  EXPECT_NEAR(vec[0], 1, 1e-5);
  EXPECT_NEAR(vec[1], -1, 1e-5);
  EXPECT_NEAR(vec[2], 2, 1e-5);
  EXPECT_NEAR(vec[3], -2, 1e-5);
}

TEST(DropOutLayer, get_layer_name) {
  EXPECT_EQ(DropOutLayer::get_name(), "DropOut layer");
}