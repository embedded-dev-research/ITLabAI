#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/ReduceLayer.hpp"
#include "layers_oneDNN/ReduceLayer.hpp"

using namespace it_lab_ai;

TEST(reducelayer_onednn, sum_2d_float_keepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {3.0F, 7.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, sum_2d_float_nokeepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 0, {1});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {3.0F, 7.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, sum_2d_int_keepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  std::vector<int> input_data = {1, 2, 3, 4};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {3, 7};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(reducelayer_onednn, mean_2d_float_keepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMean, 1, {0});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  Tensor input = make_tensor(input_data, Shape({2, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {2.5F, 3.5F, 4.5F};
  Shape expected_shape = {1, 3};

  EXPECT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, mean_2d_int_keepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMean, 1, {0});

  std::vector<int> input_data = {1, 2, 3, 4, 5, 6};
  Tensor input = make_tensor(input_data, Shape({2, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {2, 3, 4};
  Shape expected_shape = {1, 3};

  EXPECT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(reducelayer_onednn, max_3d_float_keepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMax, 1, {2});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F,  5.0F,  6.0F,
                                   7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {3.0F, 6.0F, 9.0F, 12.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, min_3d_float_nokeepdims) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMin, 0, {0, 2});

  std::vector<float> input_data = {10.0F, 2.0F, 30.0F, 4.0F, 50.0F, 6.0F,
                                   70.0F, 8.0F, 90.0F, 1.0F, 11.0F, 12.0F};

  Tensor input = make_tensor(input_data, Shape({2, 2, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {2.0F, 1.0F};
  Shape expected_shape = {2};

  EXPECT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, sum_multiple_axes) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 0, {0, 2});

  std::vector<float> input_data(2 * 3 * 4);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i + 1);
  }

  Tensor input = make_tensor(input_data, Shape({2, 3, 4}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  std::vector<float> expected = {68.0F, 100.0F, 132.0F};

  EXPECT_EQ(output_shape, Shape({3}));
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, mean_4d_float) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMean, 1, {1, 3});

  std::vector<float> input_data(2 * 3 * 4 * 5);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i + 1);
  }

  Tensor input = make_tensor(input_data, Shape({2, 3, 4, 5}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 1, 4, 1}));
}

TEST(reducelayer_onednn, negative_axes) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 0, {-1});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  Tensor input = make_tensor(input_data, Shape({2, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {6.0F, 15.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(reducelayer_onednn, all_axes) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 0, {});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {10.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  EXPECT_NEAR(output_data[0], expected[0], 1e-5);
}

TEST(reducelayer_onednn, compare_with_naive_reduction) {
  ReduceLayerOneDnn onednn_layer(ReduceLayer::Operation::kSum, 1, {1});
  ReduceLayer naive_layer(ReduceLayer::Operation::kSum, 1, {1});

  std::vector<float> input_data(16);
  for (size_t i = 0; i < 16; i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input_tensor = make_tensor(input_data, Shape({4, 4}));

  Tensor onednn_output;
  std::vector<Tensor> onednn_in{input_tensor};
  std::vector<Tensor> onednn_out{onednn_output};
  onednn_layer.run(onednn_in, onednn_out);
  auto onednn_result = *onednn_out[0].as<float>();

  Tensor naive_output;
  std::vector<Tensor> naive_in{input_tensor};
  std::vector<Tensor> naive_out{naive_output};
  naive_layer.run(naive_in, naive_out);
  auto naive_result = *naive_out[0].as<float>();

  ASSERT_EQ(onednn_result.size(), naive_result.size());
  for (size_t i = 0; i < onednn_result.size(); i++) {
    EXPECT_NEAR(onednn_result[i], naive_result[i], 1e-5);
  }
}

TEST(reducelayer_onednn, invalid_input_tensors) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  Tensor input1 = make_tensor<float>({1.0F, 2.0F});
  Tensor input2 = make_tensor<float>({3.0F, 4.0F});
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(reducelayer_onednn, scalar_input) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {});

  Tensor input = make_tensor<float>({1.0F});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(reducelayer_onednn, invalid_axis) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {5});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(reducelayer_onednn, reinitialization_different_types) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  {
    Tensor input = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    Tensor input = make_tensor<int>({1, 2, 3, 4}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<int>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    Tensor input = make_tensor<float>({5.0F, 6.0F, 7.0F, 8.0F}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }
}

TEST(reducelayer_onednn, different_shapes_same_layer) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  {
    Tensor input = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }

  {
    std::vector<float> input_data(12);
    for (size_t i = 0; i < 12; i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({3, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 3);
  }

  {
    Tensor input = make_tensor<float>({5.0F, 6.0F, 7.0F, 8.0F}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 2);
  }
}

TEST(reducelayer_onednn, set_parameters_after_creation) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  {
    Tensor input = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};
    layer.run(in, out);
  }

  layer.set_axes({0});
  layer.set_keepdims(0);
  layer.set_operation(ReduceLayer::Operation::kMean);

  {
    Tensor input =
        make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape({2, 3}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }
}

TEST(reducelayer_onednn, high_dimensional_tensors) {
  {
    ReduceLayerOneDnn layer(ReduceLayer::Operation::kMax, 1, {2, 3});

    std::vector<float> input_data(2 * 3 * 4 * 5);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({2, 3, 4, 5}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({2, 3, 1, 1}));
  }
}

TEST(reducelayer_onednn, mult_operation_int) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kMult, 1, {1});

  std::vector<int> input_data = {1, 2, 3, 4};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(reducelayer_onednn, duplicate_axes) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1, 1, 1});

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  EXPECT_EQ(result.size(), 2);
}

TEST(reducelayer_onednn, different_input_dimensions) {
  {
    ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 0, {0});

    std::vector<float> input_data(3 * 4 * 5);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({3, 4, 5}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({4, 5}));
  }

  {
    ReduceLayerOneDnn layer(ReduceLayer::Operation::kMean, 1, {1, 2});

    std::vector<float> input_data(2 * 3 * 4);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({2, 3, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({2, 1, 1}));
  }
}

TEST(reducelayer_onednn, large_tensor_performance) {
  ReduceLayerOneDnn layer(ReduceLayer::Operation::kSum, 1, {1});

  const size_t size = 1024;
  std::vector<float> input_data(size * size);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i % 100);
  }

  Tensor input = make_tensor(input_data, Shape({size, size}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  EXPECT_EQ(result.size(), size);
}
