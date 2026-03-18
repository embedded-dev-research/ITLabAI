#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"
#include "layers_oneDNN/EWLayer.hpp"

using namespace it_lab_ai;

TEST(ewlayer_onednn, supported_functions_check) {
  EXPECT_TRUE(EwLayerOneDnn::is_function_supported("relu"));
  EXPECT_TRUE(EwLayerOneDnn::is_function_supported("tanh"));
  EXPECT_TRUE(EwLayerOneDnn::is_function_supported("sigmoid"));
  EXPECT_TRUE(EwLayerOneDnn::is_function_supported("linear"));

  EXPECT_FALSE(EwLayerOneDnn::is_function_supported("sin"));
  EXPECT_FALSE(EwLayerOneDnn::is_function_supported("minus"));
  EXPECT_FALSE(EwLayerOneDnn::is_function_supported("nonexistent"));
}

TEST(ewlayer_onednn, relu_float) {
  EwLayerOneDnn layer("relu");

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

TEST(ewlayer_onednn, relu_int) {
  EwLayerOneDnn layer("relu");

  Tensor input = make_tensor<int>({1, -1, 2, -2, 0, -5});
  Tensor output;
  std::vector<int> expected = {1, 0, 2, 0, 0, 0};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(ewlayer_onednn, linear_float) {
  EwLayerOneDnn layer("linear", 2.0f, 0.0f);

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

TEST(ewlayer_onednn, linear_int) {
  EwLayerOneDnn layer("linear", 2.0f, 1.0f);

  Tensor input = make_tensor<int>({1, -1, 2, -5, 0});
  Tensor output;
  std::vector<int> expected = {3, -1, 5, -9, 1};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(ewlayer_onednn, linear_with_bias_float) {
  EwLayerOneDnn layer("linear", 1.0f, -1.0f);

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
  EwLayerOneDnn layer("tanh");

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
  EwLayerOneDnn layer("sigmoid");

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

  EwLayerOneDnn layer("relu");

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

TEST(ewlayer_onednn, multidim_tensor_relu_int) {
  Shape shape({2, 2, 2});

  EwLayerOneDnn layer("relu");

  std::vector<int> input_data = {1, -1, 2, -2, 0, -3, 4, -4};
  Tensor input = make_tensor(input_data, shape);
  Tensor output;
  std::vector<int> expected = {1, 0, 2, 0, 0, 0, 4, 0};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(ewlayer_onednn, compare_with_naive_relu) {
  EwLayerOneDnn onednn_layer("relu");

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

TEST(ewlayer_onednn, multiple_input_tensors) {
  EwLayerOneDnn layer("relu");

  Tensor input1 = make_tensor<float>({1.0F, 2.0F});
  Tensor input2 = make_tensor<float>({3.0F, 4.0F});
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(ewlayer_onednn, unsupported_tensor_dimensionality1) {
  EwLayerOneDnn layer("relu");

  Shape shape_6d({2, 3, 4, 5, 6, 7});
  std::vector<float> data_6d(2 * 3 * 4 * 5 * 6 * 7, 1.0f);
  Tensor input = make_tensor(data_6d, shape_6d);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::invalid_argument);
}

TEST(ewlayer_onednn, empty_input_tensor) {
  EwLayerOneDnn layer("relu");

  Tensor input = make_tensor<float>({});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  EXPECT_NO_THROW({ layer.run(in, out); });
}

TEST(ewlayer_onednn, invalid_function_algorithm_mapping) {
  EwLayerOneDnn layer("relu");
  EXPECT_THROW(
      {
        EwLayerOneDnn invalid_layer("invalid_function_123");
        Tensor input = make_tensor<float>({1.0F});
        Tensor output;
        std::vector<Tensor> in{input};
        std::vector<Tensor> out{output};
        invalid_layer.run(in, out);
      },
      std::invalid_argument);
}

TEST(ewlayer_onednn, initialization_failure_propagation) {
  EwLayerOneDnn layer("relu");

  Shape shape_7d({2, 2, 2, 2, 2, 2, 2});
  std::vector<float> data_7d(128, 1.0f);
  Tensor input = make_tensor(data_7d, shape_7d);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  try {
    layer.run(in, out);
    FAIL() << "Expected std::invalid_argument exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("dimensionality"), std::string::npos);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument exception";
  }
}

TEST(ewlayer_onednn, various_dimensions) {
  {
    EwLayerOneDnn layer("relu");
    Tensor input = make_tensor<float>({1.0F, -1.0F, 0.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 3);
  }

  {
    EwLayerOneDnn layer("relu");
    Shape shape2d({2, 3});
    std::vector<float> data2d = {1, -1, 2, -2, 0, 3};
    Tensor input = make_tensor(data2d, shape2d);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }

  {
    EwLayerOneDnn layer("relu");
    Shape shape3d({2, 2, 2});
    std::vector<float> data3d(8, 1.0f);
    Tensor input = make_tensor(data3d, shape3d);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }

  {
    EwLayerOneDnn layer("relu");
    Shape shape4d({1, 2, 2, 2});
    std::vector<float> data4d(8, -1.0f);
    Tensor input = make_tensor(data4d, shape4d);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    for (auto val : result) {
      EXPECT_EQ(val, 0.0f);
    }
  }

  {
    EwLayerOneDnn layer("relu");
    Shape shape5d({1, 1, 2, 2, 2});
    std::vector<float> data5d(8, 2.0f);
    Tensor input = make_tensor(data5d, shape5d);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }
}

TEST(ewlayer_onednn, linear_various_alpha_beta) {
  {
    EwLayerOneDnn layer("linear", 0.0f, 0.0f);
    Tensor input = make_tensor<float>({1.0F, 2.0F, -1.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    for (auto val : result) {
      EXPECT_NEAR(val, 0.0f, 1e-5);
    }
  }

  {
    EwLayerOneDnn layer("linear", 1.0f, 0.0f);
    Tensor input = make_tensor<float>({1.0F, 2.0F, -1.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    auto input_data = *input.as<float>();
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], input_data[i], 1e-5);
    }
  }

  {
    EwLayerOneDnn layer("linear", 0.0f, 5.0f);
    Tensor input = make_tensor<float>({1.0F, 2.0F, -1.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    for (auto val : result) {
      EXPECT_NEAR(val, 5.0f, 1e-5);
    }
  }

  {
    EwLayerOneDnn layer("linear", -2.0f, 3.0f);
    Tensor input = make_tensor<float>({1.0F, 2.0F, -1.0F});
    Tensor output;
    std::vector<float> expected = {1.0f, -1.0f, 5.0f};

    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], expected[i], 1e-5);
    }
  }
}

TEST(ewlayer_onednn, reinitialization_for_int) {
  EwLayerOneDnn layer("relu");

  {
    Tensor input = make_tensor<float>({1.0F, -1.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto float_result = *out[0].as<float>();
    EXPECT_NEAR(float_result[0], 1.0f, 1e-5);
    EXPECT_NEAR(float_result[1], 0.0f, 1e-5);
  }

  {
    Tensor input = make_tensor<int>({1, -1});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto int_result = *out[0].as<int>();
    EXPECT_EQ(int_result[0], 1);
    EXPECT_EQ(int_result[1], 0);
  }

  {
    Tensor input = make_tensor<float>({2.0F, -2.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto float_result = *out[0].as<float>();
    EXPECT_NEAR(float_result[0], 2.0f, 1e-5);
    EXPECT_NEAR(float_result[1], 0.0f, 1e-5);
  }
}

TEST(ewlayer_onednn, edge_case_dimensions) {
  {
    EwLayerOneDnn layer("relu");
    Shape large_shape({1000});
    std::vector<float> large_data(1000);
    for (size_t i = 0; i < large_data.size(); i++) {
      large_data[i] = static_cast<float>(i) - 500.0f;
    }

    Tensor input = make_tensor(large_data, large_shape);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1000);
  }

  {
    EwLayerOneDnn layer("relu");
    Tensor input = make_tensor<float>({5.0F});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
    EXPECT_NEAR(result[0], 5.0f, 1e-5);
  }
}

TEST(ewlayer_onednn, all_functions_int) {
  std::vector<std::pair<std::string, std::vector<int>>> test_cases = {
      {"relu", {1, 0, 2, 0}},
      {"linear", {3, -1, 5, -9}},
  };

  for (const auto& [func, expected] : test_cases) {
    EwLayerOneDnn layer(func, 2.0f, 1.0f);

    std::vector<int> input_data;
    if (func == "relu") {
      input_data = {1, -1, 2, -2};
    } else if (func == "linear") {
      input_data = {1, -1, 2, -5};
    }

    Tensor input = make_tensor<int>(input_data);
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<int>();

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_EQ(result[i], expected[i])
          << "Function: " << func << ", index: " << i;
    }
  }
}

TEST(ewlayer_onednn, unsupported_tensor_dimensionality) {
  EwLayerOneDnn layer("relu");
  Shape shape6d({2, 2, 2, 2, 2, 2});
  std::vector<float> data6d(64, 1.0f);
  Tensor input = make_tensor(data6d, shape6d);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);

  try {
    layer.run(in, out);
    FAIL() << "Expected std::invalid_argument exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("Unsupported tensor dimensionality"),
              std::string::npos);
    EXPECT_NE(std::string(e.what()).find("6"), std::string::npos);
  }
}

TEST(ewlayer_onednn, unsupported_function_algorithm) {
  EwLayerOneDnn layer("relu");
  EwLayerOneDnn invalid_layer("invalid_func_123", 1.0f, 0.0f);

  Tensor input = make_tensor<float>({1.0F});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(invalid_layer.run(in, out), std::invalid_argument);

  try {
    invalid_layer.run(in, out);
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("Unsupported function for oneDNN"),
              std::string::npos);
    EXPECT_NE(std::string(e.what()).find("invalid_func_123"),
              std::string::npos);
  }
}
