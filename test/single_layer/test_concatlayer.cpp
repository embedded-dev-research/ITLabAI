#include <vector>

#include "gtest/gtest.h"
#include "layers/ConCatLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

class ConcatLayerTests : public ::testing::Test {
 protected:
  void SetUp() override {
    data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    data_int = {1, 2, 3, 4};
  }

  std::vector<float> data1;
  std::vector<float> data2;
  std::vector<int> data_int;
};

TEST_F(ConcatLayerTests, ConcatSameShapeFloatAxis0) {
  ConcatLayer layer;
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
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

TEST_F(ConcatLayerTests, ConcatSameShapeIntAxis1) {
  ConcatLayer layer(1);
  Tensor input1 = make_tensor<int>(data_int, {2, 2});
  Tensor input2 = make_tensor<int>(data_int, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 4}));
  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 1);
  EXPECT_EQ((*result)[3], 2);
  EXPECT_EQ((*result)[4], 3);
  EXPECT_EQ((*result)[7], 4);
}

TEST_F(ConcatLayerTests, Concat3DTensorsAxis2) {
  ConcatLayer layer(2);
  Tensor input1 = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor input2 =
      make_tensor<float>({9, 10, 11, 12, 13, 14, 15, 16}, {2, 2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 2, 4}));
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[3], 4.0f);
  EXPECT_FLOAT_EQ((*result)[4], 5.0f);
  EXPECT_FLOAT_EQ((*result)[11], 16.0f);
}

TEST_F(ConcatLayerTests, NegativeAxis) {
  ConcatLayer layer(-1);
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 4}));
}

TEST_F(ConcatLayerTests, DynamicAxis) {
  ConcatLayer layer(1);
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 4}));
}

TEST_F(ConcatLayerTests, IncompatibleShapes) {
  ConcatLayer layer(0);
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  EXPECT_THROW(layer.run({input1, input2}, output), std::runtime_error);
}

TEST_F(ConcatLayerTests, LayerName) {
  EXPECT_EQ(ConcatLayer::get_name(), "ConcatLayer");
}

TEST_F(ConcatLayerTests, EmptyTensors) {
  ConcatLayer layer(0);
  Tensor empty1({}, Type::kFloat);
  Tensor empty2({}, Type::kFloat);
  Tensor output;

  EXPECT_NO_THROW(layer.run({empty1, empty2}, output));
}

TEST_F(ConcatLayerTests, ConcatMultipleTensors) {
  ConcatLayer layer(0);
  Tensor input1 = make_tensor<float>({1, 2}, {2});
  Tensor input2 = make_tensor<float>({3, 4}, {2});
  Tensor input3 = make_tensor<float>({5, 6}, {2});
  Tensor output;

  layer.run({input1, input2, input3}, output);

  ASSERT_EQ(output.get_shape(), Shape({6}));
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[3], 4.0f);
  EXPECT_FLOAT_EQ((*result)[5], 6.0f);
}

TEST_F(ConcatLayerTests, ConcatDifferentTypes) {
  ConcatLayer layer(0);
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<int>(data_int, {4});
  Tensor output;

  EXPECT_THROW(layer.run({input1, input2}, output), std::runtime_error);
}

TEST_F(ConcatLayerTests, ConcatResNetStyle) {
  ConcatLayer layer(1);
  Tensor input1 = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 2, 2});
  Tensor input2 =
      make_tensor<float>({9, 10, 11, 12, 13, 14, 15, 16}, {1, 2, 2, 2});
  Tensor output;

  layer.run({input1, input2}, output);

  ASSERT_EQ(output.get_shape(), Shape({1, 4, 2, 2}));
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[8], 9.0f);
  EXPECT_FLOAT_EQ((*result)[15], 16.0f);
}