#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/MatmulLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(MatmulLayerTest, DotProduct1D1D) {
  Tensor input1 = make_tensor<float>({1, 2, 3}, {3});
  Tensor input2 = make_tensor<float>({4, 5, 6}, {3});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({}));
  EXPECT_FLOAT_EQ(out[0].get<float>({}), 32.0f);
}

TEST(MatmulLayerTest, VectorMatrixMultiplication1D2D) {
  Tensor input1 = make_tensor<float>({1, 2, 3}, {3});
  Tensor input2 = make_tensor<float>({4, 5, 6, 7, 8, 9}, {3, 2});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 40.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 46.0f);
}

TEST(MatmulLayerTest, MatrixVectorMultiplication2D1D) {
  Tensor input1 = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  Tensor input2 = make_tensor<float>({5, 6}, {2});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 17.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 39.0f);
}

TEST(MatmulLayerTest, BatchMatrixMultiplicationWithBroadcasting) {
  std::vector<float> a_data(1 * 3 * 3 * 4, 1.0f);
  std::vector<float> b_data(1 * 3 * 4 * 3, 2.0f);

  Tensor input1 = make_tensor<float>(a_data, {1, 3, 3, 4});
  Tensor input2 = make_tensor<float>(b_data, {1, 3, 4, 3});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 3, 4, 4}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 2, 2, 1}), 6.0f);
}

TEST(MatmulLayerTest, DifferentBatchDimensionsBroadcasting) {
  std::vector<float> a_data(3 * 4 * 3 * 4, 1.0f);
  std::vector<float> b_data(3 * 4 * 4 * 3, 1.0f);

  Tensor input1 = make_tensor<float>(a_data, {3, 4, 3, 4});
  Tensor input2 = make_tensor<float>(b_data, {3, 4, 4, 3});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({3, 4, 4, 4}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({2, 3, 1, 2}), 3.0f);
}

TEST(MatmulLayerTest, ComplexBroadcastingExample) {
  std::vector<float> a_data;
  std::vector<float> b_data;

  for (size_t i = 0; i < 4 * 2 * 5 * 4; ++i) a_data.push_back(1.0f);
  for (size_t i = 0; i < 4 * 2 * 4 * 5; ++i) b_data.push_back(1.0f);

  Tensor input1 = make_tensor<float>(a_data, {4, 2, 5, 4});
  Tensor input2 = make_tensor<float>(b_data, {4, 2, 4, 5});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({4, 2, 5, 5}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({3, 1, 2, 4}), 4.0f);
}

TEST(MatmulLayerTest, SingleElementTensors) {
  Tensor input1 = make_tensor<float>({5.0f}, {1});
  Tensor input2 = make_tensor<float>({6.0f}, {1});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({}));
  EXPECT_FLOAT_EQ(out[0].get<float>({}), 30.0f);
}

TEST(MatmulLayerTest, MixedDimensionsComplexCase) {
  std::vector<float> a_data;
  for (size_t i = 0; i < 3 * 4 * 5; ++i)
    a_data.push_back(static_cast<float>(i % 5 + 1));
  std::vector<float> b_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Tensor input1 = make_tensor<float>(a_data, {3, 4, 5});
  Tensor input2 = make_tensor<float>(b_data, {5});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({3, 4}));

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 55.0f);
}

TEST(MatmulLayerTest, IncompatibleBroadcasting) {
  Tensor input1 =
      make_tensor<float>(std::vector<float>(2 * 3 * 4, 1.0f), {2, 3, 4});
  Tensor input2 =
      make_tensor<float>(std::vector<float>(4 * 5 * 6, 1.0f), {4, 5, 6});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(MatmulLayerTest, Original4DCase) {
  std::vector<float> a_data(1 * 6 * 64 * 49, 1.0f);
  std::vector<float> b_data(1 * 6 * 49 * 49, 1.0f);

  Tensor input1 = make_tensor<float>(a_data, {1, 6, 64, 49});
  Tensor input2 = make_tensor<float>(b_data, {1, 6, 49, 49});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 6, 64, 49}));

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 49.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 5, 63, 48}), 49.0f);
}

TEST(MatmulLayerTest, Specific4DCase_49x32_and_32x49) {
  std::vector<float> a_data(1 * 6 * 49 * 32);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = 1.0f;
  }

  std::vector<float> b_data(1 * 6 * 32 * 49);
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = 1.0f;
  }

  Tensor input1 = make_tensor<float>(a_data, {1, 6, 49, 32});
  Tensor input2 = make_tensor<float>(b_data, {1, 6, 32, 49});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 6, 49, 49}));

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 48}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 48, 0}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 48, 48}), 32.0f);

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 5, 0, 0}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 5, 0, 48}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 5, 48, 0}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 5, 48, 48}), 32.0f);

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 2, 10, 25}), 32.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 3, 40, 15}), 32.0f);
}

TEST(MatmulLayerTest, Specific4DCase_WithDifferentValues) {
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,

                               7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> b_data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,

                               7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  Tensor input1 = make_tensor<float>(a_data, {1, 2, 3, 2});
  Tensor input2 = make_tensor<float>(b_data, {1, 2, 2, 3});
  MatmulLayer layer;
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({1, 2, 3, 3}));

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 9.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 2}), 15.0f);

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 0}), 19.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 1}), 26.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 2}), 33.0f);

  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 2, 0}), 29.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 2, 1}), 40.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 2, 2}), 51.0f);
}