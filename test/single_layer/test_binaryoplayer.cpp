#include <vector>

#include "gtest/gtest.h"
#include "layers/BinaryOpLayer.hpp"
#include "layers/Tensor.hpp"

using namespace itlab_2023;

class BinaryOpLayerMulTests : public ::testing::Test {
 protected:
  void SetUp() override {
    data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    data2 = {2.0f, 3.0f, 4.0f, 5.0f};
    data_int = {1, 2, 3, 4};
    scalar = make_tensor<float>({2.0f});
    scalar_int = make_tensor<int>({2});
  }

  std::vector<float> data1;
  std::vector<float> data2;
  std::vector<int> data_int;
  Tensor scalar;
  Tensor scalar_int;
};

TEST_F(BinaryOpLayerMulTests, MulSameShapeFloat) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 6.0f);
  EXPECT_FLOAT_EQ((*result)[2], 12.0f);
  EXPECT_FLOAT_EQ((*result)[3], 20.0f);
}

TEST_F(BinaryOpLayerMulTests, MulSameShapeInt) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<int>(data_int, {2, 2});
  Tensor input2 = make_tensor<int>(data_int, {2, 2});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 1);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 9);
  EXPECT_EQ((*result)[3], 16);
}

TEST_F(BinaryOpLayerMulTests, MulSameShapeIntResNet1) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<int>({1, 2, 64, 64, 64}, {5});
  Tensor input2 = make_tensor<int>({1, 2, 64, 1, 1}, {5});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 1);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 4096);
  EXPECT_EQ((*result)[3], 64);
  EXPECT_EQ((*result)[4], 64);
}

TEST_F(BinaryOpLayerMulTests, MulWithScalarFloat) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input = make_tensor<float>(data1, {2, 2});
  Tensor output;

  layer.run(input, scalar, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);
}

TEST_F(BinaryOpLayerMulTests, MulWithScalarInt) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input = make_tensor<int>(data_int, {2, 2});
  Tensor output;

  layer.run(input, scalar_int, output);

  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 2);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 6);
  EXPECT_EQ((*result)[3], 8);
}

TEST_F(BinaryOpLayerMulTests, BroadcastingTest1) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<float>({1.0f, 2.0f}, {2, 1});
  Tensor input2 = make_tensor<float>({3.0f, 4.0f}, {1, 2});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 3.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);
}

TEST_F(BinaryOpLayerMulTests, Broadcasting3D) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 1, 3});
  Tensor input2 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3, 1});
  Tensor output;

  layer.run(input1, input2, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 3, 3}));
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[1], 2.0f);
  EXPECT_FLOAT_EQ((*result)[2], 3.0f);
  EXPECT_FLOAT_EQ((*result)[3], 2.0f);
  EXPECT_FLOAT_EQ((*result)[4], 4.0f);
  EXPECT_FLOAT_EQ((*result)[5], 6.0f);
  EXPECT_FLOAT_EQ((*result)[12], 20.0f);
  EXPECT_FLOAT_EQ((*result)[13], 25.0f);
  EXPECT_FLOAT_EQ((*result)[14], 30.0f);
  EXPECT_FLOAT_EQ((*result)[15], 24.0f);
  EXPECT_FLOAT_EQ((*result)[16], 30.0f);
  EXPECT_FLOAT_EQ((*result)[17], 36.0f);
}

TEST_F(BinaryOpLayerMulTests, BroadcastingDifferentRanks) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<float>({1.0f, 2.0f, 3.0f}, {3});
  Tensor input2 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 1, 3});
  Tensor output;

  layer.run(input1, input2, output);

  ASSERT_EQ(output.get_shape(), Shape({2, 1, 3}));
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 9.0f);
  EXPECT_FLOAT_EQ((*result)[3], 4.0f);
}

TEST_F(BinaryOpLayerMulTests, IncompatibleShapes) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  EXPECT_THROW(layer.run(input1, input2, output), std::runtime_error);
}

TEST_F(BinaryOpLayerMulTests, LayerName) {
  EXPECT_EQ(BinaryOpLayer::get_name(), "Binary Operation Layer");
}

TEST_F(BinaryOpLayerMulTests, EmptyTensors) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::MUL);
  Tensor empty1({}, Type::kFloat);
  Tensor empty2({}, Type::kFloat);
  Tensor output;

  EXPECT_NO_THROW(layer.run(empty1, empty2, output));
}

TEST_F(BinaryOpLayerMulTests, BroadcastingTestAdd) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::ADD);

  Tensor input1 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1, 1, 1});

  Tensor input2 = make_tensor<float>(
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      {5, 4, 1, 1});

  Tensor output;
  layer.run(input1, input2, output);

  ASSERT_EQ(output.get_shape(), Shape({5, 4, 1, 1}));

  auto* result = output.as<float>();

  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 3.0f);
  EXPECT_FLOAT_EQ((*result)[4], 7.0f);
  EXPECT_FLOAT_EQ((*result)[5], 8.0f);
}

TEST_F(BinaryOpLayerMulTests, BroadcastingTestSubGooglNet) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::SUB);
  Tensor input1 = make_tensor<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},
      {1, 2, 3, 3});
  Tensor output;

  layer.run(input1, scalar, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[2], 1.0f);
  EXPECT_FLOAT_EQ((*result)[5], 4.0f);
  EXPECT_FLOAT_EQ((*result)[12], 11.0f);
  EXPECT_FLOAT_EQ((*result)[17], 16.0f);
}