#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/ConcatLayer.hpp"
#include "layers_oneDNN/ConcatLayer.hpp"

using namespace it_lab_ai;

TEST(concatlayer_onednn, concat_basic_axis0_2d_float) {
  ConcatLayerOneDnn layer(0);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F,
                                 5.0F, 6.0F, 7.0F, 8.0F};
  Shape expected_shape({4, 2});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_basic_axis1_2d_float) {
  ConcatLayerOneDnn layer(1);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {1.0F, 2.0F, 5.0F, 6.0F,
                                 3.0F, 4.0F, 7.0F, 8.0F};
  Shape expected_shape({2, 4});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_basic_axis0_2d_int) {
  ConcatLayerOneDnn layer(0);

  std::vector<int> a_data = {1, 2, 3, 4};
  std::vector<int> b_data = {5, 6, 7, 8};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape expected_shape({4, 2});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(concatlayer_onednn, concat_basic_axis1_2d_int) {
  ConcatLayerOneDnn layer(1);

  std::vector<int> a_data = {1, 2, 3, 4};
  std::vector<int> b_data = {5, 6, 7, 8};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {1, 2, 5, 6, 3, 4, 7, 8};
  Shape expected_shape({2, 4});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(concatlayer_onednn, concat_three_tensors_axis0_float) {
  ConcatLayerOneDnn layer(0);

  std::vector<float> a_data = {1.0F, 2.0F};
  std::vector<float> b_data = {3.0F, 4.0F};
  std::vector<float> c_data = {5.0F, 6.0F};
  Tensor a = make_tensor(a_data, Shape({2, 1}));
  Tensor b = make_tensor(b_data, Shape({2, 1}));
  Tensor c = make_tensor(c_data, Shape({2, 1}));
  Tensor output;

  std::vector<Tensor> in{a, b, c};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  Shape expected_shape({6, 1});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_three_tensors_axis1_float) {
  ConcatLayerOneDnn layer(1);

  std::vector<float> a_data = {1.0F, 2.0F};
  std::vector<float> b_data = {3.0F, 4.0F};
  std::vector<float> c_data = {5.0F, 6.0F};
  Tensor a = make_tensor(a_data, Shape({1, 2}));
  Tensor b = make_tensor(b_data, Shape({1, 2}));
  Tensor c = make_tensor(c_data, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b, c};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  Shape expected_shape({1, 6});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_3d_tensors_axis0) {
  ConcatLayerOneDnn layer(0);

  std::vector<float> a_data(2 * 3 * 4);
  std::vector<float> b_data(2 * 3 * 4);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 100);
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4}));
  Tensor b = make_tensor(b_data, Shape({2, 3, 4}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({4, 3, 4}));
  EXPECT_EQ(output_data.size(), a_data.size() + b_data.size());

  for (size_t i = 0; i < a_data.size(); i++) {
    EXPECT_NEAR(output_data[i], a_data[i], 1e-5);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    EXPECT_NEAR(output_data[i + a_data.size()], b_data[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_3d_tensors_axis1) {
  ConcatLayerOneDnn layer(1);

  std::vector<float> a_data(2 * 3 * 4);
  std::vector<float> b_data(2 * 3 * 4);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 1000);
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4}));
  Tensor b = make_tensor(b_data, Shape({2, 3, 4}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 6, 4}));
  EXPECT_EQ(output_data.size(), a_data.size() + b_data.size());

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        size_t idx_a = i * 3 * 4 + j * 4 + k;
        size_t idx_out_a = i * 6 * 4 + j * 4 + k;
        EXPECT_NEAR(output_data[idx_out_a], a_data[idx_a], 1e-5);
      }
    }
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        size_t idx_b = i * 3 * 4 + j * 4 + k;
        size_t idx_out_b = i * 6 * 4 + (j + 3) * 4 + k;
        EXPECT_NEAR(output_data[idx_out_b], b_data[idx_b], 1e-5);
      }
    }
  }
}

TEST(concatlayer_onednn, concat_4d_tensors_axis3) {
  ConcatLayerOneDnn layer(3);

  std::vector<float> a_data(2 * 3 * 4 * 5);
  std::vector<float> b_data(2 * 3 * 4 * 3);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    b_data[i] = static_cast<float>(i + 1000);
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4, 5}));
  Tensor b = make_tensor(b_data, Shape({2, 3, 4, 3}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 3, 4, 8}));
  EXPECT_EQ(output_data.size(), a_data.size() + b_data.size());

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        for (size_t l = 0; l < 5; l++) {
          size_t idx_a = i * 3 * 4 * 5 + j * 4 * 5 + k * 5 + l;
          size_t idx_out_a = i * 3 * 4 * 8 + j * 4 * 8 + k * 8 + l;
          EXPECT_NEAR(output_data[idx_out_a], a_data[idx_a], 1e-5);
        }
        for (size_t l = 0; l < 3; l++) {
          size_t idx_b = i * 3 * 4 * 3 + j * 4 * 3 + k * 3 + l;
          size_t idx_out_b = i * 3 * 4 * 8 + j * 4 * 8 + k * 8 + l + 5;
          EXPECT_NEAR(output_data[idx_out_b], b_data[idx_b], 1e-5);
        }
      }
    }
  }
}

TEST(concatlayer_onednn, concat_negative_axis) {
  ConcatLayerOneDnn layer(-1);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {1.0F, 2.0F, 5.0F, 6.0F,
                                 3.0F, 4.0F, 7.0F, 8.0F};
  Shape expected_shape({2, 4});

  ASSERT_EQ(out[0].get_shape(), expected_shape);
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(concatlayer_onednn, concat_single_tensor) {
  ConcatLayerOneDnn layer(0);

  std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();

  ASSERT_EQ(out[0].get_shape(), input.get_shape());
  ASSERT_EQ(output_data.size(), data.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], data[i], 1e-5);
  }
}

TEST(concatlayer_onednn, compare_with_naive_implementation) {
  ConcatLayerOneDnn onednn_layer(1);
  ConcatLayer naive_layer(1);

  std::vector<float> a_data(12);
  std::vector<float> b_data(12);
  for (size_t i = 0; i < 12; i++) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i * 2);
  }

  Tensor a_tensor = make_tensor(a_data, Shape({3, 4}));
  Tensor b_tensor = make_tensor(b_data, Shape({3, 4}));

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

TEST(concatlayer_onednn, incompatible_types) {
  ConcatLayerOneDnn layer(0);

  Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
  Tensor b = make_tensor<int>({3, 4}, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(concatlayer_onednn, incompatible_ranks) {
  ConcatLayerOneDnn layer(0);

  Tensor a = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}));
  Tensor b = make_tensor<float>(
      {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}, Shape({2, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(concatlayer_onednn, invalid_axis_out_of_range) {
  ConcatLayerOneDnn layer(2);

  std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
  Tensor a = make_tensor(a_data, Shape({2, 2}));
  Tensor b = make_tensor(b_data, Shape({2, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(concatlayer_onednn, reinitialization_different_types) {
  ConcatLayerOneDnn layer(0);

  {
    Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({3.0F, 4.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(out[0].get_shape(), Shape({2, 2}));
  }

  {
    Tensor a = make_tensor<int>({1, 2}, Shape({1, 2}));
    Tensor b = make_tensor<int>({3, 4}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<int>();
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(out[0].get_shape(), Shape({2, 2}));
  }

  {
    Tensor a = make_tensor<float>({5.0F, 6.0F}, Shape({1, 2}));
    Tensor b = make_tensor<float>({7.0F, 8.0F}, Shape({1, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(out[0].get_shape(), Shape({2, 2}));
  }
}

TEST(concatlayer_onednn, different_axis_same_layer) {
  ConcatLayerOneDnn layer(0);

  {
    std::vector<float> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> b_data = {5.0F, 6.0F, 7.0F, 8.0F};
    Tensor a = make_tensor(a_data, Shape({2, 2}));
    Tensor b = make_tensor(b_data, Shape({2, 2}));
    Tensor output;
    std::vector<Tensor> in{a, b};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    EXPECT_EQ(out[0].get_shape(), Shape({4, 2}));
  }
}

TEST(concatlayer_onednn, high_dimensional_tensors) {
  ConcatLayerOneDnn layer(2);

  std::vector<float> a_data(2 * 3 * 4 * 5 * 6);
  std::vector<float> b_data(2 * 3 * 7 * 5 * 6);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    b_data[i] = static_cast<float>(i + 1000);
  }

  Tensor a = make_tensor(a_data, Shape({2, 3, 4, 5, 6}));
  Tensor b = make_tensor(b_data, Shape({2, 3, 7, 5, 6}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  auto output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({2, 3, 11, 5, 6}));
  EXPECT_EQ(result.size(), a_data.size() + b_data.size());
}

TEST(concatlayer_onednn, large_tensors_performance) {
  ConcatLayerOneDnn layer(1);

  const size_t batch = 4;
  const size_t channels = 64;
  const size_t height = 32;
  const size_t width = 32;

  std::vector<float> a_data(batch * channels * height * width);
  std::vector<float> b_data(batch * channels * height * width);
  for (size_t i = 0; i < a_data.size(); i++) {
    a_data[i] = static_cast<float>(i % 100);
    b_data[i] = static_cast<float>((i + 1) % 100);
  }

  Tensor a = make_tensor(a_data, Shape({batch, channels, height, width}));
  Tensor b = make_tensor(b_data, Shape({batch, channels, height, width}));
  Tensor output;
  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto result = *out[0].as<float>();
  auto output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({batch, channels * 2, height, width}));
  EXPECT_EQ(result.size(), a_data.size() + b_data.size());
}

TEST(concatlayer_onednn, edge_cases_zero_dimension) {
  ConcatLayerOneDnn layer(0);

  std::vector<float> a_data = {};
  std::vector<float> b_data = {1.0F, 2.0F};

  Tensor a = make_tensor(a_data, Shape({0, 2}));
  Tensor b = make_tensor(b_data, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  try {
    layer.run(in, out);
    auto result = *out[0].as<float>();
    EXPECT_EQ(out[0].get_shape(), Shape({1, 2}));
    EXPECT_EQ(result.size(), 2);
    EXPECT_NEAR(result[0], 1.0F, 1e-5);
    EXPECT_NEAR(result[1], 2.0F, 1e-5);
  } catch (const std::exception& e) {
    GTEST_LOG_(INFO) << "Zero dimension test skipped: " << e.what();
  }
}

TEST(concatlayer_onednn, mixed_float_int_not_allowed) {
  ConcatLayerOneDnn layer(0);

  Tensor a = make_tensor<float>({1.0F, 2.0F}, Shape({1, 2}));
  Tensor b = make_tensor<int>({3, 4}, Shape({1, 2}));
  Tensor output;

  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}
