#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/BinaryOpLayer.hpp"
#include "layers_oneDNN/BinaryOpLayer.hpp"

using namespace it_lab_ai;

TEST(binaryoplayer_onednn, add_basic_float) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {6.0F, 8.0F, 10.0F, 12.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, mul_basic_float) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {2.0F, 3.0F, 4.0F, 5.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {2.0F, 6.0F, 12.0F, 20.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, add_basic_int) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<int> a_data = {1, 2, 3, 4};
  std::vector<int> b_data = {5, 6, 7, 8};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {6, 8, 10, 12};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(binaryoplayer_onednn, mul_basic_int) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  std::vector<int> a_data = {1, 2, 3, 4};
  std::vector<int> b_data = {2, 3, 4, 5};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {2, 6, 12, 20};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(binaryoplayer_onednn, broadcast_scalar_float) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {10.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({1, 1}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {11.0F, 12.0F, 13.0F, 14.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, broadcast_row_vector) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<float> b_data = {2.0F, 3.0F};
  Tensor a = make_tensor(a_data, Shape({3, 2}));
  Tensor b = make_tensor(b_data, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {2.0F, 6.0F, 6.0F, 12.0F, 10.0F, 18.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, broadcast_column_vector) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<float> b_data = {10.0F, 20.0F, 30.0F};
  Tensor a = make_tensor(a_data, Shape({3, 2}));
  Tensor b = make_tensor(b_data, Shape({3, 1}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {11.0F, 12.0F, 23.0F, 24.0F, 35.0F, 36.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, different_shapes_3d) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  std::vector<float> a_data(2 * 3 * 4);
  std::vector<float> b_data(1 * 3 * 4);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    b_data[i] = 2.0F;
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4}));
  Tensor b = make_tensor(b_data, Shape({1, 3, 4}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 3, 4}));
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], a_data[i] * 2.0F, 1e-5);
  }
}

TEST(binaryoplayer_onednn, different_shapes_4d) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data(2 * 3 * 4 * 5);
  std::vector<float> b_data(1 * 1 * 4 * 5);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    b_data[i] = 100.0F;
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4, 5}));
  Tensor b = make_tensor(b_data, Shape({1, 1, 4, 5}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 3, 4, 5}));
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], a_data[i] + 100.0F, 1e-5);
  }
}

TEST(binaryoplayer_onednn, compare_with_naive_implementation) {
  BinaryOpLayerOneDnn onednn_layer(BinaryOpLayer::Operation::kAdd);
  BinaryOpLayer naive_layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data(16);
  std::vector<float> b_data(16);
  for (size_t i = 0; i < 16; i++) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i * 2);
  }

  Tensor a_tensor = make_tensor(a_data, Shape({4, 4}));
  Tensor b_tensor = make_tensor(b_data, Shape({4, 4}));

  Tensor onednn_output;
  std::vector<Tensor> onednn_in{a_tensor, b_tensor};
  std::vector<Tensor> onednn_out{onednn_output};
  onednn_layer.run(onednn_in, onednn_out);
  auto onednn_result = *onednn_out[0].as<float>();

  Tensor naive_output;
  std::vector<Tensor> naive_in{a_tensor, b_tensor};
  std::vector<Tensor> naive_out{naive_output};
  naive_layer.run(naive_in, naive_out);
  auto naive_result = *naive_out[0].as<float>();

  ASSERT_EQ(onednn_result.size(), naive_result.size());
  for (size_t i = 0; i < onednn_result.size(); i++) {
    EXPECT_NEAR(onednn_result[i], naive_result[i], 1e-5);
  }
}

TEST(binaryoplayer_onednn, invalid_input_tensors) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  Tensor a = make_tensor<float>({1.0F, 2.0F});
  Tensor b = make_tensor<float>({3.0F, 4.0F});
  Tensor c = make_tensor<float>({5.0F, 6.0F});
  Tensor output;

  std::vector<Tensor> in{a, b, c};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(binaryoplayer_onednn, incompatible_types) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
  Tensor b = make_tensor<int>({3, 4}, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(binaryoplayer_onednn, incompatible_shapes) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  Tensor a = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}));
  Tensor b =
      make_tensor<float>({5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F}, Shape({2, 3}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(binaryoplayer_onednn, reinitialization_different_types) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  {
    Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({3.0F, 4.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    Tensor a = make_tensor<int>({1, 2}, Shape({1, 2}));
    Tensor b = make_tensor<int>({3, 4}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<int>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    Tensor a = make_tensor<float>({5.0F, 6.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({7.0F, 8.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }
}

TEST(binaryoplayer_onednn, different_shapes_same_layer) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kMul);

  {
    Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({3.0F, 4.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    std::vector<float> a_data(12);
    std::vector<float> b_data(12);
    for (size_t i = 0; i < 12; i++) {
      a_data[i] = static_cast<float>(i);
      b_data[i] = static_cast<float>(i + 1);
    }

    Tensor a = make_tensor(a_data, Shape({3, 4}));
    Tensor b = make_tensor(b_data, Shape({3, 4}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 12);
  }

  {
    Tensor a = make_tensor<float>({5.0F, 6.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({7.0F, 8.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }
}

TEST(binaryoplayer_onednn, set_operation_after_creation) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  {
    Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({3.0F, 4.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};
    layer.run(in, out);
  }

  layer.set_operation(BinaryOpLayer::Operation::kMul);

  {
    Tensor a = make_tensor<float>({2.0F, 3.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({4.0F, 5.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    std::vector<float> expected = {8.0F, 15.0F};
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], expected[i], 1e-5);
    }
  }
}

TEST(binaryoplayer_onednn, high_dimensional_tensors) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  std::vector<float> a_data(2 * 3 * 4 * 5);
  std::vector<float> b_data(1 * 1 * 1 * 5);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    b_data[i] = static_cast<float>(i * 10);
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4, 5}));
  Tensor b = make_tensor(b_data, Shape({1, 1, 1, 5}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  auto output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape, Shape({2, 3, 4, 5}));
  EXPECT_EQ(result.size(), a_data.size());
}

TEST(binaryoplayer_onednn, large_tensor_performance) {
  BinaryOpLayerOneDnn layer(BinaryOpLayer::Operation::kAdd);

  const size_t size = 512;
  std::vector<float> a_data(size * size);
  std::vector<float> b_data(size * size);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i % 100);
    b_data[i] = static_cast<float>((i + 1) % 100);
  }

  Tensor a = make_tensor(a_data, Shape({size, size}));
  Tensor b = make_tensor(b_data, Shape({size, size}));
  Tensor output;
  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  EXPECT_EQ(result.size(), size * size);
}
