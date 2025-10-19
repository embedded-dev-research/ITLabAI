#include <vector>

#include "gtest/gtest.h"
#include "layers/FlattenLayer.hpp"

using namespace it_lab_ai;

TEST(flattenlayer, flatten_with_axis_1) {
  FlattenLayer layer(1);
  Shape sh({2, 3, 4});
  Tensor input =
      make_tensor<int>({1, -1, 2, -2, 3, -3, 4,  -4,  5,  -5,  6,  -6,
                        7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12},
                       sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 2);
  EXPECT_EQ(out[0].get_shape()[0], 2);
  EXPECT_EQ(out[0].get_shape()[1], 12);
}

TEST(flattenlayer, flatten_with_axis_0) {
  FlattenLayer layer(0);
  Shape sh({2, 3});
  Tensor input =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 1);
  EXPECT_EQ(out[0].get_shape()[0], 6);
}

TEST(flattenlayer, flatten_with_different_axis_values) {
  std::vector<int> axis_values = {0, 1, 2, -1};

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
    if (axis == 0) {
      EXPECT_EQ(out[0].get_shape().dims(), 1);
      EXPECT_EQ(out[0].get_shape()[0], 24);
    } else if (axis == 1) {
      EXPECT_EQ(out[0].get_shape().dims(), 2);
      EXPECT_EQ(out[0].get_shape()[0], 2);
      EXPECT_EQ(out[0].get_shape()[1], 12);
    } else if (axis == 2 || axis == -1) {
      EXPECT_EQ(out[0].get_shape().dims(), 3);
      EXPECT_EQ(out[0].get_shape()[0], 2);
      EXPECT_EQ(out[0].get_shape()[1], 3);
      EXPECT_EQ(out[0].get_shape()[2], 4);
    }
  }
}

TEST(flattenlayer, flatten_3d_tensor_with_axis_1) {
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
  EXPECT_EQ(out[0].get_shape().dims(), 2);
  EXPECT_EQ(out[0].get_shape()[0], 2);
  EXPECT_EQ(out[0].get_shape()[1], 12);
}

TEST(flattenlayer, flatten_4d_tensor_with_axis_2) {
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
  EXPECT_EQ(out[0].get_shape().dims(), 3);
  EXPECT_EQ(out[0].get_shape()[0], 2);
  EXPECT_EQ(out[0].get_shape()[1], 2);
  EXPECT_EQ(out[0].get_shape()[2], 6);
}

TEST(flattenlayer, flatten_with_negative_axis) {
  FlattenLayer layer(-2);
  Shape sh({2, 3, 4});

  std::vector<int> input_data(24);
  for (size_t i = 0; i < 24; i++) {
    input_data[i] = static_cast<int>(i);
  }

  Tensor input = make_tensor<int>(input_data, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  EXPECT_EQ(out[0].get_shape().dims(), 2);
  EXPECT_EQ(out[0].get_shape()[0], 2);
  EXPECT_EQ(out[0].get_shape()[1], 12);
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

TEST(flattenlayer, MultipleInputTensorsThrowsError) {
  FlattenLayer layer;
  Shape sh({2, 3});
  Tensor input1 =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor input2 =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(flattenlayer, InvalidAxisValueThrowsError) {
  FlattenLayer layer(5);
  Shape sh({2, 3});
  Tensor input =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(flattenlayer, NegativeAxisOutOfRangeThrowsError) {
  FlattenLayer layer(-5);
  Shape sh({2, 3});
  Tensor input =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(flattenlayer, AxisEqualToShapeDimsThrowsError) {
  FlattenLayer layer(2);
  Shape sh({2, 3});
  Tensor input =
      make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(flattenlayer, ValidAxisWithSupportedTypes) {
  std::vector<int> axis_values = {0, 1, -1, -2};

  for (int axis : axis_values) {
    FlattenLayer layer(axis);
    Shape sh({2, 3, 4});
    size_t total_size = sh.count();

    std::vector<float> float_data(total_size);
    std::vector<int> int_data(total_size);
    for (size_t i = 0; i < total_size; i++) {
      float_data[i] = static_cast<float>(i);
      int_data[i] = static_cast<int>(i);
    }

    Tensor float_input = make_tensor<float>(float_data, sh);
    Tensor int_input = make_tensor<int>(int_data, sh);
    Tensor output;

    std::vector<Tensor> float_in{float_input};
    std::vector<Tensor> int_in{int_input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(float_in, out));
    EXPECT_NO_THROW(layer.run(int_in, out));
  }
}

TEST(flattenlayer, EmptyInputThrowsError) {
  FlattenLayer layer;
  std::vector<Tensor> in;
  std::vector<Tensor> out(1);

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}
