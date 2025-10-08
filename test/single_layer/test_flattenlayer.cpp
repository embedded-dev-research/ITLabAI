#include <vector>

#include "gtest/gtest.h"
#include "layers/FlattenLayer.hpp"

using namespace it_lab_ai;

TEST(flattenlayer, new_flattenlayer_can_flatten_int) {
  FlattenLayer layer;
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 4);
}

TEST(flattenlayer, new_flattenlayer_can_flatten_float) {
  FlattenLayer layer;
  Shape sh({2, 2});
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 4);
}

TEST(flattenlayer, new_flattenlayer_can_flatten_float_reorder) {
  FlattenLayer layer1;
  FlattenLayer layer2({1, 2, 3, 0});  // NCHW -> CHWN
  FlattenLayer layer3({0, 2, 3, 1});  // NCHW -> NHWC
  Shape sh({2, 2, 2, 3});
  std::vector<float> input_vec(sh.count());
  for (size_t i = 0; i < sh.count(); i++) {
    input_vec[i] = static_cast<float>(i);
  }
  std::vector<float> expected_2 = {0.0f, 12.0f, 1.0f,  13.0f, 2.0f,  14.0f,
                                   3.0f, 15.0f, 4.0f,  16.0f, 5.0f,  17.0f,
                                   6.0f, 18.0f, 7.0f,  19.0f, 8.0f,  20.0f,
                                   9.0f, 21.0f, 10.0f, 22.0f, 11.0f, 23.0f};
  std::vector<float> expected_3 = {0.0f,  6.0f,  1.0f,  7.0f,  2.0f,  8.0f,
                                   3.0f,  9.0f,  4.0f,  10.0f, 5.0f,  11.0f,
                                   12.0f, 18.0f, 13.0f, 19.0f, 14.0f, 20.0f,
                                   15.0f, 21.0f, 16.0f, 22.0f, 17.0f, 23.0f};
  Tensor input = make_tensor<float>(input_vec, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer1.run(in, out);
  EXPECT_EQ(*out[0].as<float>(), input_vec);
  layer2.run(in, out);
  EXPECT_EQ(*out[0].as<float>(), expected_2);
  layer3.run(in, out);
  EXPECT_EQ(*out[0].as<float>(), expected_3);
}

TEST(flattenlayer, new_flattenlayer_can_flatten_int_reorder) {
  FlattenLayer layer1;
  FlattenLayer layer2({1, 2, 3, 0});  // NCHW -> CHWN
  FlattenLayer layer3({0, 2, 3, 1});  // NCHW -> NHWC
  Shape sh({2, 2, 2, 3});
  std::vector<int> input_vec(sh.count());
  for (size_t i = 0; i < sh.count(); i++) {
    input_vec[i] = static_cast<int>(i);
  }
  std::vector<int> expected_2 = {0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                                 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23};
  std::vector<int> expected_3 = {0,  6,  1,  7,  2,  8,  3,  9,
                                 4,  10, 5,  11, 12, 18, 13, 19,
                                 14, 20, 15, 21, 16, 22, 17, 23};
  Tensor input = make_tensor<int>(input_vec, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer1.run(in, out);
  EXPECT_EQ(*out[0].as<int>(), input_vec);
  layer2.run(in, out);
  EXPECT_EQ(*out[0].as<int>(), expected_2);
  layer3.run(in, out);
  EXPECT_EQ(*out[0].as<int>(), expected_3);
}
