#include <vector>

#include "layers/Tensor.hpp"
#include "gtest/gtest.h"
#include "layers/MulLayer.hpp"

using namespace itlab_2023;

class MulLayerTests : public ::testing::Test {
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

// Тест умножения тензоров одинаковой формы (float)
TEST_F(MulLayerTests, MulSameShapeFloat) {
  MulLayer layer;
  Tensor input1 = make_tensor<float>(data1, {2, 2});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);   // 1*2
  EXPECT_FLOAT_EQ((*result)[1], 6.0f);   // 2*3
  EXPECT_FLOAT_EQ((*result)[2], 12.0f);  // 3*4
  EXPECT_FLOAT_EQ((*result)[3], 20.0f);  // 4*5
}

// Тест умножения тензоров одинаковой формы (int)
TEST_F(MulLayerTests, MulSameShapeInt) {
  MulLayer layer;
  Tensor input1 = make_tensor<int>(data_int, {2, 2});
  Tensor input2 = make_tensor<int>(data_int, {2, 2});
  Tensor output;

  layer.run(input1, input2, output);

  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 1);   // 1*1
  EXPECT_EQ((*result)[1], 4);   // 2*2
  EXPECT_EQ((*result)[2], 9);   // 3*3
  EXPECT_EQ((*result)[3], 16);  // 4*4
}

// Тест умножения тензора на скаляр (float)
TEST_F(MulLayerTests, MulWithScalarFloat) {
  MulLayer layer;
  Tensor input = make_tensor<float>(data1, {2, 2});
  Tensor output;

  layer.run(input, scalar, output);

  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 2.0f);  // 1*2
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);  // 2*2
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);  // 3*2
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);  // 4*2
}

// Тест умножения тензора на скаляр (int)
TEST_F(MulLayerTests, MulWithScalarInt) {
  MulLayer layer;
  Tensor input = make_tensor<int>(data_int, {2, 2});
  Tensor output;

  layer.run(input, scalar_int, output);

  auto* result = output.as<int>();
  EXPECT_EQ((*result)[0], 2);  // 1*2
  EXPECT_EQ((*result)[1], 4);  // 2*2
  EXPECT_EQ((*result)[2], 6);  // 3*2
  EXPECT_EQ((*result)[3], 8);  // 4*2
}

// Тест broadcasting (расширения размерностей)
TEST_F(MulLayerTests, BroadcastingTest) {
  MulLayer layer;
  Tensor input1 = make_tensor<float>({1.0f, 2.0f}, {2, 1});  // shape [2,1]
  Tensor input2 = make_tensor<float>({3.0f, 4.0f}, {1, 2});  // shape [1,2]
  Tensor output;

  layer.run(input1, input2, output);

  // Ожидаемый результат после broadcasting:
  // [[1*3, 1*4], [2*3, 2*4]] = [[3,4], [6,8]]
  auto* result = output.as<float>();
  EXPECT_FLOAT_EQ((*result)[0], 3.0f);
  EXPECT_FLOAT_EQ((*result)[1], 4.0f);
  EXPECT_FLOAT_EQ((*result)[2], 6.0f);
  EXPECT_FLOAT_EQ((*result)[3], 8.0f);
}

// Тест на несовместимые формы
TEST_F(MulLayerTests, IncompatibleShapes) {
  MulLayer layer;
  Tensor input1 = make_tensor<float>(data1, {4});
  Tensor input2 = make_tensor<float>(data2, {2, 2});
  Tensor output;

  EXPECT_THROW(layer.run(input1, input2, output), std::runtime_error);
}

// Тест имени слоя
TEST_F(MulLayerTests, LayerName) {
  EXPECT_EQ(MulLayer::get_name(), "Element-wise Multiplication Layer");
}

// Тест умножения пустых тензоров
TEST_F(MulLayerTests, EmptyTensors) {
  MulLayer layer;
  Tensor empty1({}, Type::kFloat);
  Tensor empty2({}, Type::kFloat);
  Tensor output;

  EXPECT_NO_THROW(layer.run(empty1, empty2, output));
}
