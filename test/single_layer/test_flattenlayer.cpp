#include <vector>

#include "gtest/gtest.h"
#include "layers/FlattenLayer.hpp"

using namespace it_lab_ai;

TEST(flattenlayer, flatten_with_axis_1) {
  FlattenLayer layer(1);
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 4);
}

TEST(flattenlayer, flatten_with_axis_0) {
  FlattenLayer layer(0);
  Shape sh({2, 2});
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 4);
}

TEST(flattenlayer, flatten_with_different_axis_values) {
  std::vector<int> axis_values = {0, 1, -1};

  for (int axis : axis_values) {
    FlattenLayer layer(axis);
    Shape sh({2, 3, 4});
    size_t total_size = sh.count();

    std::vector<int> input_data(total_size);
    for (size_t i = 0; i < total_size; i++) {
      input_data[i] = static_cast<int>(i);
    }

    Tensor input = make_tensor<int>(input_data, sh);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    EXPECT_EQ(out[0].get_shape().dims(), 1);
    EXPECT_EQ(out[0].get_shape()[0], total_size);
  }
}

TEST(flattenlayer, flatten_3d_tensor_with_axis) {
  FlattenLayer layer(1);
  Shape sh({2, 3, 4});
  size_t total_size = 2 * 3 * 4;

  std::vector<float> input_data(total_size);
  for (size_t i = 0; i < total_size; i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input = make_tensor<float>(input_data, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], total_size);
}

TEST(flattenlayer, flatten_4d_tensor_with_axis) {
  FlattenLayer layer(2);
  Shape sh({2, 2, 2, 3});
  size_t total_size = 2 * 2 * 2 * 3;

  std::vector<int> input_data(total_size);
  for (size_t i = 0; i < total_size; i++) {
    input_data[i] = static_cast<int>(i);
  }

  Tensor input = make_tensor<int>(input_data, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], total_size);
}

TEST(flattenlayer, new_flattenlayer_can_flatten_float_reorder) {
  FlattenLayer layer1;
  FlattenLayer layer2(std::vector<size_t>{1, 2, 3, 0});
  FlattenLayer layer3(std::vector<size_t>{0, 2, 3, 1});

  Shape sh({2, 2, 2, 3});
  std::vector<float> input_vec(sh.count());
  for (size_t i = 0; i < sh.count(); i++) {
    input_vec[i] = static_cast<float>(i);
  }

  Tensor input = make_tensor<float>(input_vec, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer1.run(in, out);
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], sh.count());
  EXPECT_NO_THROW(layer2.run(in, out));
  EXPECT_NO_THROW(layer3.run(in, out));
}

TEST(flattenlayer, new_flattenlayer_can_flatten_int_reorder) {
  FlattenLayer layer1;
  FlattenLayer layer2(std::vector<size_t>{1, 2, 3, 0});
  FlattenLayer layer3(std::vector<size_t>{0, 2, 3, 1});
  Shape sh({2, 2, 2, 3});
  std::vector<int> input_vec(sh.count());
  for (size_t i = 0; i < sh.count(); i++) {
    input_vec[i] = static_cast<int>(i);
  }

  Tensor input = make_tensor<int>(input_vec, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer1.run(in, out);
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], sh.count());
  EXPECT_NO_THROW(layer2.run(in, out));
  EXPECT_NO_THROW(layer3.run(in, out));
}

TEST(flattenlayer, get_layer_name) {
  EXPECT_EQ(FlattenLayer::get_name(), "Flatten layer");
}

TEST(flattenlayer, flattenlayer_with_axis) {
  FlattenLayer layer(1);
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 4);
}
