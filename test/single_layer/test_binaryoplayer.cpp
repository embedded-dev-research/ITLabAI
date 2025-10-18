#include <vector>

#include "gtest/gtest.h"
#include "layers/BinaryOpLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

class BinaryOpLayerTests : public ::testing::Test {
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

TEST_F(BinaryOpLayerTests, MulSameShapeFloat) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  auto* result = output[0].as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 6.0f);
  EXPECT_FLOAT_EQ((*result)[2], 12.0f);
  EXPECT_FLOAT_EQ((*result)[3], 20.0f);
}

TEST_F(BinaryOpLayerTests, MulSameShapeInt) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<int>(data_int, {2, 2});
  Tensor input2 = make_tensor<int>(data_int, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  auto* result = output[0].as<int>();
  EXPECT_EQ((*result)[0], 1);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 9);
  EXPECT_EQ((*result)[3], 16);
}

TEST_F(BinaryOpLayerTests, MulSameShapeIntResNet1) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<int>({1, 2, 64, 64, 64}, {5});
  Tensor input2 = make_tensor<int>({1, 2, 64, 1, 1}, {5});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  auto* result = output[0].as<int>();
  EXPECT_EQ((*result)[0], 1);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 4096);
  EXPECT_EQ((*result)[3], 64);
  EXPECT_EQ((*result)[4], 64);
}

TEST_F(BinaryOpLayerTests, MulWithScalarFloat) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input = make_tensor<float>(data1, {2, 2});
  std::vector<Tensor> in{input, scalar};
  std::vector<Tensor> output{input};

  layer.run(in, output);

  auto* result = output[0].as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);
}

TEST_F(BinaryOpLayerTests, MulWithScalarInt) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input = make_tensor<int>(data_int, {2, 2});
  std::vector<Tensor> in{input, scalar_int};
  std::vector<Tensor> output{input};

  layer.run(in, output);

  auto* result = output[0].as<int>();
  EXPECT_EQ((*result)[0], 2);
  EXPECT_EQ((*result)[1], 4);
  EXPECT_EQ((*result)[2], 6);
  EXPECT_EQ((*result)[3], 8);
}

TEST_F(BinaryOpLayerTests, BroadcastingTest1) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>({1.0f, 2.0f}, {2, 1});
  Tensor input2 = make_tensor<float>({3.0f, 4.0f}, {1, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  auto* result = output[0].as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 3.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);
}

TEST_F(BinaryOpLayerTests, Broadcasting3D) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 1, 3});
  Tensor input2 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3, 1});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  ASSERT_EQ(output[0].get_shape(), Shape({2, 3, 3}));
  auto* result = output[0].as<float>();
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

TEST_F(BinaryOpLayerTests, BroadcastingDifferentRanks) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>({1.0f, 2.0f, 3.0f}, {3});
  Tensor input2 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 1, 3});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  ASSERT_EQ(output[0].get_shape(), Shape({2, 1, 3}));
  auto* result = output[0].as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 1.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 9.0f);
  EXPECT_FLOAT_EQ((*result)[3], 4.0f);
}

TEST_F(BinaryOpLayerTests, IncompatibleShapes) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  EXPECT_THROW(layer.run(in, output), std::runtime_error);
}

TEST_F(BinaryOpLayerTests, EmptyTensors) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor empty1({}, Type::kFloat);
  Tensor empty2({}, Type::kFloat);
  std::vector<Tensor> in{empty1, empty2};
  std::vector<Tensor> output{empty1};

  EXPECT_NO_THROW(layer.run(in, output));
}

TEST_F(BinaryOpLayerTests, BroadcastingTestAdd) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);

  Tensor input1 =
      make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1, 1, 1});

  Tensor input2 = make_tensor<float>(
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      {5, 4, 1, 1});

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};
  layer.run(in, output);

  ASSERT_EQ(output[0].get_shape(), Shape({5, 4, 1, 1}));

  auto* result = output[0].as<float>();

  EXPECT_FLOAT_EQ((*result)[0], 2.0f);
  EXPECT_FLOAT_EQ((*result)[1], 3.0f);
  EXPECT_FLOAT_EQ((*result)[4], 7.0f);
  EXPECT_FLOAT_EQ((*result)[5], 8.0f);
}

TEST_F(BinaryOpLayerTests, BroadcastingTestSubGooglNet) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kSub);
  Tensor input1 = make_tensor<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},
      {1, 2, 3, 3});
  std::vector<Tensor> in{input1, scalar};
  std::vector<Tensor> output{input1};

  layer.run(in, output);

  auto* result = output[0].as<float>();
  EXPECT_FLOAT_EQ((*result)[2], 1.0f);
  EXPECT_FLOAT_EQ((*result)[5], 4.0f);
  EXPECT_FLOAT_EQ((*result)[12], 11.0f);
  EXPECT_FLOAT_EQ((*result)[17], 16.0f);
}

TEST_F(BinaryOpLayerTests, IncompatibleInput) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>(data1, {4});
  std::vector<Tensor> in{input1};
  std::vector<Tensor> output{input1};

  EXPECT_THROW(layer.run(in, output), std::runtime_error);
}

TEST_F(BinaryOpLayerTests, IncompatibleInputType) {
  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<int>(data_int, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};

  EXPECT_THROW(layer.run(in, output), std::runtime_error);
}