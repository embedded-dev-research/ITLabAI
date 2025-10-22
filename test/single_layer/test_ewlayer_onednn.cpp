#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"
#include "layers_oneDNN/EWLayer_oneDNN.hpp"

using namespace it_lab_ai;

TEST(ewlayer_onednn, supported_functions_check) {
  EXPECT_TRUE(EWLayer_oneDNN::is_function_supported("relu"));
  EXPECT_TRUE(EWLayer_oneDNN::is_function_supported("tanh"));
  EXPECT_TRUE(EWLayer_oneDNN::is_function_supported("sigmoid"));
  EXPECT_TRUE(EWLayer_oneDNN::is_function_supported("linear"));

  EXPECT_FALSE(EWLayer_oneDNN::is_function_supported("sin"));
  EXPECT_FALSE(EWLayer_oneDNN::is_function_supported("minus"));
  EXPECT_FALSE(EWLayer_oneDNN::is_function_supported("nonexistent"));
}

TEST(ewlayer_onednn, relu_float) {
  EWLayer_oneDNN layer("relu");

  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> expected = {1.0F, 0.0F, 2.0F, 0.0F};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(ewlayer_onednn, linear_float) {
  EWLayer_oneDNN layer("linear", 2.0f, 0.0f);

  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> expected = {2.0F, -2.0F, 4.0F, -10.0F};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(ewlayer_onednn, linear_with_bias_float) {
  EWLayer_oneDNN layer("linear", 1.0f, -1.0f);

  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> expected = {0.0F, -2.0F, 1.0F, -6.0F};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(ewlayer_onednn, tanh_float) {
  EWLayer_oneDNN layer("tanh");

  Tensor input = make_tensor<float>({0.0F, 1.0F, -1.0F, 2.0F});
  Tensor output;
  std::vector<float> expected;

  std::vector<float> input_data = *input.as<float>();
  for (auto val : input_data) {
    expected.push_back(std::tanh(val));
  }

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(ewlayer_onednn, sigmoid_float) {
  EWLayer_oneDNN layer("sigmoid");

  Tensor input = make_tensor<float>({0.0F, 1.0F, -1.0F, 2.0F});
  Tensor output;
  std::vector<float> expected;

  std::vector<float> input_data = *input.as<float>();
  for (auto val : input_data) {
    expected.push_back(1.0f / (1.0f + std::exp(-val)));
  }

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(ewlayer_onednn, multidim_tensor_relu) {
  Shape shape({1, 3, 2, 2});

  EWLayer_oneDNN layer("relu");

  std::vector<float> input_data(1 * 3 * 2 * 2);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i) - 2.0f;
  }

  Tensor input = make_tensor(input_data, shape);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  ASSERT_EQ(output_data.size(), input_data.size());

  for (size_t i = 0; i < output_data.size(); i++) {
    float expected = std::max(0.0f, input_data[i]);
    EXPECT_NEAR(output_data[i], expected, 1e-5);
  }
}

TEST(ewlayer_onednn, compare_with_naive_relu) {
  EWLayer_oneDNN onednn_layer("relu");

  EWLayer naive_layer("relu");

  std::vector<float> input_data(100);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i) - 50.0f;
  }

  Tensor input_tensor = make_tensor<float>(input_data);

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
