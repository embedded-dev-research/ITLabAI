#include <vector>

#include "gtest/gtest.h"
#include "layers/FCLayer.hpp"

using namespace it_lab_ai;

class FCTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, std::vector<double>, Shape,
                     std::vector<double>, std::vector<double> > > {};
// 1) input; 2) weights; 3) weights_shape; 4) bias; 5) expected_output.

TEST_P(FCTestsParameterized, fc_layer_works_correctly) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  std::vector<double> weights = std::get<1>(data);
  Shape wshape = std::get<2>(data);
  std::vector<double> bias = std::get<3>(data);
  FCLayerImpl<double> layer(weights, wshape, bias);
  std::vector<double> output = layer.run(input);
  std::vector<double> expected_output = std::get<4>(data);
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], expected_output[i], 1e-5);
  }
}

std::vector<double> basic_weights1 = {2.0, 0.1, 0.0, 1.5, 1.9, 5.5};

std::vector<double> basic_weights2 = {4.1, -2.3, 6.0, 9.0, 3.0, -3.4,
                                      7.0, 0.0,  1.9, 8.0, 8.0, -1.0};
std::vector<double> basic_bias1 = {0.5, 0.5, 1.0};
std::vector<double> basic_bias2 = {2.0, 2.0, 2.0};
std::vector<double> basic_bias1_corrected = {0.5, 0.5, 1.0};
std::vector<double> basic_bias2_corrected = {2.0, 2.0, 2.0};
INSTANTIATE_TEST_SUITE_P(
    fc_layer_tests, FCTestsParameterized,
    ::testing::Values(
        std::make_tuple(std::vector<double>({1.0, 2.0}), basic_weights1,
                        Shape({2, 3}), basic_bias1,
                        std::vector<double>({5.5, 4.4, 12.0})),

        std::make_tuple(std::vector<double>({0.5, 0.0}), basic_weights1,
                        Shape({2, 3}), basic_bias1,
                        std::vector<double>({1.5, 0.55, 1.0})),

        std::make_tuple(std::vector<double>({1.0, -1.0, 1.0, -1.0}),
                        basic_weights2, Shape({4, 3}), basic_bias2,
                        std::vector<double>({-3.9, -11.3, 14.3})),

        std::make_tuple(std::vector<double>({1.0, 0.0, 1.0, 0.0}),
                        basic_weights2, Shape({4, 3}), basic_bias2,
                        std::vector<double>({13.1, -0.3, 9.9}))));

TEST(fclayer, throws_when_empty_weights) {
  const std::vector<double> a1;
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  ASSERT_ANY_THROW(FCLayerImpl<double> layer(a1, wshape, bias));
}
TEST(fclayer, throws_when_empty_bias) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias;
  ASSERT_ANY_THROW(FCLayerImpl<double> layer(a1, wshape, bias));
}

TEST(fclayer, matvecmul_works) {
  std::vector<int> mat = {2, 4, 2, 3};
  std::vector<int> vec = {1, 2};
  Shape mat_shape({2, 2});
  std::vector<int> true_res = {6, 10};
  std::vector<int> res = mat_vec_mul(mat, mat_shape, vec);
  EXPECT_EQ(res, true_res);
}
TEST(fclayer, set_get_bias_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5};
  FCLayerImpl<double> layer(a1, wshape, bias);

  for (size_t i = 0; i < bias.size(); i++) {
    EXPECT_NEAR(layer.get_bias(i), bias[i], 1e-5);
  }

  for (size_t i = 0; i < bias.size(); i++) {
    layer.set_bias(i, static_cast<double>(i));
    EXPECT_NEAR(layer.get_bias(i), static_cast<double>(i), 1e-5);
  }
}

TEST(fclayer, set_get_weight_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape wshape({3, 3});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  ASSERT_ANY_THROW(layer.get_weight(4, 0));
  ASSERT_ANY_THROW(layer.get_weight(0, 4));
  ASSERT_ANY_THROW(layer.set_weight(4, 0, 1.3));
  ASSERT_ANY_THROW(layer.set_weight(0, 4, 1.3));
}
TEST(fclayer, set_get_bias_throws_when_out_of_range) {
  const std::vector<double> a1 = {2.0, 1.5, 3.5, 0.1, 1.9, 2.6, 0.0, 5.5, 1.7};
  Shape wshape({3, 3});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  ASSERT_ANY_THROW(layer.get_bias(4));
  ASSERT_ANY_THROW(layer.set_bias(4, 1.3));
}

TEST(fclayer, get_dims_returns_correctly) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5};
  FCLayerImpl<double> layer(a1, wshape, bias);

  EXPECT_EQ(layer.get_dims().first[0], 2);
  EXPECT_EQ(layer.get_dims().second[0], 3);
}

TEST(fclayer, matvecmul_throws_when_not_matrix) {
  std::vector<int> mat = {2, 4, 2, 4, 1, 3, 5, 7};
  std::vector<int> vec = {1, 2};
  Shape mat_shape({2, 2, 2});
  ASSERT_ANY_THROW(mat_vec_mul(mat, mat_shape, vec));
}

TEST(fclayer, new_fc_layer_can_run_float) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> a2 = {10.2F, 3.5F, 17.7F};

  Tensor weights = make_tensor<float>(a1, {2, 3});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor output;
  FCLayer layer(weights, bias);
  std::vector<Tensor> in{make_tensor<float>({2.0F, 3.0F})};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<float> result = *out[0].as<float>();
  ASSERT_EQ(result.size(), a2.size());

  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_NEAR(result[i], a2[i], 1e-5);
  }
}

TEST(fclayer, new_fc_layer_can_run_int) {
  const std::vector<int> a1 = {2, 1, 0, 2, 0, 5};
  const std::vector<int> a2 = {10, 2, 16};
  Tensor weights = make_tensor<int>(a1, {2, 3});
  Tensor bias = make_tensor<int>({0, 0, 1});
  Tensor output;
  FCLayer layer(weights, bias);
  std::vector<Tensor> in{make_tensor<int>({2, 3})};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();
  ASSERT_EQ(result.size(), a2.size());

  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_EQ(result[i], a2[i]);
  }
}

TEST(fclayer, new_fc_layer_throws_when_big_input) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  FCLayer layer;
  std::vector<Tensor> in{make_tensor<float>({2.0F, 3.0F, 4.0F})};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(fclayer, new_fc_layer_throws_with_incorrect_bias_type) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<int>({2, 5, 6});
  FCLayer layer;
  std::vector<Tensor> in{make_tensor<float>({2.0F, 3.0F})};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(fclayer, IncompatibleInput) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<int>({2, 5, 6});
  FCLayer layer;
  std::vector<Tensor> in{make_tensor<float>({2.0F, 3.0F}),
                         make_tensor<float>({2.0F, 3.0F})};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(fclayer, new_fc_layer_throws_with_incorrect_input_type) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({2, 5, 6});
  FCLayer layer;
  std::vector<Tensor> in{make_tensor<int>({2, 3})};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(fclayer, InvalidWeightsSizeZeroOutput) {
  std::vector<float> weightsvec = {};
  Shape weights_shape({10, 0});
  Tensor weights = make_tensor(weightsvec, weights_shape);

  std::vector<float> biasvec = {};
  Tensor bias = make_tensor(biasvec, Shape({0}));

  std::vector<float> input_vec(10, 1.0f);
  Tensor input = make_tensor(input_vec, Shape({10}));

  std::vector<float> output_vec(0, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({0}));

  FCLayer layer(weights, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(fclayer, new_fc_bias_and_weights_not_same) {
  const std::vector<int> a1 = {2, 1, 0, 2, 0, 5};
  const std::vector<int> a2 = {10, 2, 16};
  Tensor weights = make_tensor<int>(a1, {2, 3});
  Tensor bias = make_tensor<float>({0, 0, 1});
  Tensor output;
  FCLayer layer(weights, bias);
  std::vector<Tensor> in{make_tensor<int>({2, 3})};
  std::vector<Tensor> out{output};
  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(fclayer, VectorSizeNotDivisibleByMatrixRows) {
  std::vector<float> weightsvec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Shape weights_shape({3, 2});
  Tensor weights = make_tensor(weightsvec, weights_shape);

  std::vector<float> biasvec = {0.1f, 0.2f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  std::vector<float> input_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  Tensor input = make_tensor(input_vec, Shape({5}));

  std::vector<float> output_vec(4, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({2, 2}));

  FCLayer layer(weights, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(fclayer, VectorSizeNotDivisibleByMatrixRowsInt) {
  std::vector<int> weightsvec = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape weights_shape({4, 2});
  Tensor weights = make_tensor(weightsvec, weights_shape);

  std::vector<int> biasvec = {1, 2};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  std::vector<int> input_vec = {1, 2, 3, 4, 5, 6, 7};
  Tensor input = make_tensor(input_vec, Shape({7}));

  std::vector<int> output_vec(4, 0);
  Tensor output = make_tensor(output_vec, Shape({2, 2}));

  FCLayer layer(weights, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(fclayer, VectorSizeDivisibleByMatrixRows) {
  std::vector<float> weightsvec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Shape weights_shape({3, 2});
  Tensor weights = make_tensor(weightsvec, weights_shape);

  std::vector<float> biasvec = {0.1f, 0.2f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  std::vector<float> input_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor input = make_tensor(input_vec, Shape({6}));

  std::vector<float> output_vec(4, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({2, 2}));

  FCLayer layer(weights, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
}

TEST(fclayer, ZeroOutputNeuronsWithNonZeroInput) {
  std::vector<float> weightsvec = {};
  Shape weights_shape({5, 0});
  Tensor weights = make_tensor(weightsvec, weights_shape);

  std::vector<float> biasvec = {};
  Tensor bias = make_tensor(biasvec, Shape({0}));

  std::vector<float> input_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  Tensor input = make_tensor(input_vec, Shape({5}));

  std::vector<float> output_vec = {};
  Tensor output = make_tensor(output_vec, Shape({0}));

  FCLayer layer(weights, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(fclayer, matvecmul_batch_processing) {
  std::vector<int> mat = {1, 2, 3, 4, 5, 6};
  Shape mat_shape({2, 3});
  std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> expected = {9, 12, 15, 19, 26, 33};

  std::vector<int> result = mat_vec_mul(mat, mat_shape, vec);
  EXPECT_EQ(result, expected);
}

TEST(fclayer, matvecmul_batch_size_3) {
  std::vector<float> mat = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape mat_shape({2, 2});
  std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> expected = {
      1.0f * 1.0f + 2.0f * 3.0f, 1.0f * 2.0f + 2.0f * 4.0f,
      3.0f * 1.0f + 4.0f * 3.0f, 3.0f * 2.0f + 4.0f * 4.0f,
      5.0f * 1.0f + 6.0f * 3.0f, 5.0f * 2.0f + 6.0f * 4.0f};
  std::vector<float> result = mat_vec_mul(mat, mat_shape, vec);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}

TEST(fclayer, matvecmul_layout_verification) {
  std::vector<int> mat = {1, 10, 2, 20, 3, 30};
  Shape mat_shape({3, 2});
  std::vector<int> vec = {1, 1, 1};
  std::vector<int> expected = {6, 60};
  std::vector<int> result = mat_vec_mul(mat, mat_shape, vec);
  EXPECT_EQ(result, expected);
}

TEST(fclayer, BatchProcessingWithBias) {
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape weights_shape({2, 2});
  std::vector<float> bias = {0.1f, 0.2f};

  FCLayerImpl<float> layer(weights, weights_shape, bias);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> output = layer.run(input);
  std::vector<float> expected = {7.1f, 10.2f, 15.1f, 22.2f};

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f);
  }
}

TEST(fclayer, BatchSize3WithBiasVerification) {
  std::vector<int> weights = {1, 2, 3, 4};
  Shape weights_shape({2, 2});
  std::vector<int> bias = {10, 20};

  FCLayerImpl<int> layer(weights, weights_shape, bias);
  std::vector<int> input = {1, 1, 2, 2, 3, 3};
  std::vector<int> output = layer.run(input);
  std::vector<int> expected = {14, 26, 18, 32, 22, 38};

  EXPECT_EQ(output, expected);
}
