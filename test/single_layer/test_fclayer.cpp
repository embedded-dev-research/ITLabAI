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

std::vector<double> basic_weights1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
std::vector<double> basic_weights2 = {4.1, 3.0, 1.9, -1.2, -2.3, -3.4,
                                      6.0, 7.0, 8.0, 9.0,  0.0,  -1.0};
std::vector<double> basic_bias1 = {0.5, 0.5, 1.0};

INSTANTIATE_TEST_SUITE_P(
    fc_layer_tests, FCTestsParameterized,
    ::testing::Values(
        std::make_tuple(std::vector<double>({1.0, 2.0}), basic_weights1,
                        Shape({3, 2}), basic_bias1,
                        std::vector<double>({5.5, 4.4, 12.0})),
        std::make_tuple(std::vector<double>({0.5, 0.0}), basic_weights1,
                        Shape({3, 2}), basic_bias1,
                        std::vector<double>({1.5, 0.55, 1.0})),
        std::make_tuple(std::vector<double>({1.0, -1.0, 1.0, -1.0}),
                        basic_weights2, Shape({3, 4}),
                        std::vector<double>({2.0, 2.0, 2.0}),
                        std::vector<double>({6.2, 2.1, 2.0})),
        std::make_tuple(std::vector<double>({1.0, 0.0, 1.0, 0.0}),
                        basic_weights2, Shape({3, 4}),
                        std::vector<double>({2.0, 2.0, 2.0}),
                        std::vector<double>({8.0, 5.7, 10.0}))));

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

TEST(fclayer, set_get_weight_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  for (size_t i = 0; i < wshape[0]; i++) {
    for (size_t j = 0; j < wshape[1]; j++) {
      EXPECT_NEAR(layer.get_weight(i, j), a1[wshape.get_index({i, j})], 1e-5);
    }
  }
  for (size_t i = 0; i < wshape[0]; i++) {
    for (size_t j = 0; j < wshape[1]; j++) {
      layer.set_weight(i, j, static_cast<double>(i + j));
      EXPECT_NEAR(layer.get_weight(i, j), static_cast<double>(i + j), 1e-5);
    }
  }
}
TEST(fclayer, set_get_bias_is_correct) {
  const std::vector<double> a1 = {2.0, 1.5, 0.1, 1.9, 0.0, 5.5};
  Shape wshape({3, 2});
  std::vector<double> bias = {0.5, 0.5, 1.0};
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
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayerImpl<double> layer(a1, wshape, bias);
  EXPECT_EQ(layer.get_dims().first[0], 3);
  EXPECT_EQ(layer.get_dims().second[0], 2);
}

TEST(fclayer, matvecmul_works) {
  std::vector<int> mat = {2, 4, 2, 3};
  std::vector<int> vec = {1, 2};
  Shape mat_shape({2, 2});
  std::vector<int> true_res = {10, 8};
  std::vector<int> res = mat_vec_mul(mat, mat_shape, vec);
  EXPECT_EQ(res, true_res);
}

TEST(fclayer, matvecmul_throws_when_not_matrix) {
  std::vector<int> mat = {2, 4, 2, 4, 1, 3, 5, 7};
  std::vector<int> vec = {1, 2};
  Shape mat_shape({2, 2, 2});
  ASSERT_ANY_THROW(mat_vec_mul(mat, mat_shape, vec));
}

TEST(fclayer, new_fc_layer_can_run_float) {
  const std::vector<float> a1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> a2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  FCLayer layer(weights, bias);
  std::vector<Tensor> in{make_tensor<float>({2.0F, 3.0F})};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], a2[i], 1e-5);
  }
}

TEST(fclayer, new_fc_layer_can_run_int) {
  const std::vector<int> a1 = {2, 1, 0, 2, 0, 5};
  const std::vector<int> a2 = {7, 6, 16};
  Tensor weights = make_tensor<int>(a1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<int>({0, 0, 1});
  FCLayer layer(weights, bias);
  std::vector<Tensor> in{make_tensor<int>({2, 3})};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < a2.size(); i++) {
    EXPECT_NEAR((*out[0].as<int>())[i], a2[i], 1e-5);
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
