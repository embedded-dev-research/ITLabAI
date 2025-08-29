#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

TEST(poolinglayer, empty_inputs1) {
  Shape inpshape = 0;
  Shape poolshape = 0;
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, empty_inputs2) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input;
  ASSERT_ANY_THROW(std::vector<double> output = a.run(input));
}

TEST(poolinglayer, empty_inputs3) {
  Shape inpshape = {3};
  Shape poolshape = {0};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, throws_when_big_input) {
  Shape inpshape = {7};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  ASSERT_ANY_THROW(a.run(input));
}

TEST(poolinglayer, tbb_pl_throws_when_big_input) {
  Shape inpshape = {7};
  Shape poolshape = {3};
  PoolingLayerImplTBB<double> a =
      PoolingLayerImplTBB<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  ASSERT_ANY_THROW(a.run(input));
}

TEST(poolinglayer, throws_when_invalid_pooling_type) {
  Shape inpshape = {7};
  Shape poolshape = {3};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "my"));
}

TEST(poolinglayer, throws_when_bigger_pooling_dims) {
  Shape inpshape = {8};
  Shape poolshape = {8, 8};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, throws_when_bigger_input_dims) {
  Shape inpshape = {2, 3, 4, 5, 6};
  Shape poolshape = {2, 2};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, pooling_throws_when_more_than_2d) {
  Shape inpshape = {4, 4, 4};
  Shape poolshape = {2, 1, 3};
  ASSERT_ANY_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, equivalent_output_when_pool_size_1) {
  Shape inpshape = {8};
  Shape poolshape = {1};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  PoolingLayerImpl<double> b =
      PoolingLayerImpl<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output_a = a.run(input);
  std::vector<double> output_b = b.run(input);
  for (size_t i = 0; i < output_a.size(); i++) {
    EXPECT_NEAR(output_a[i], input[i], 1e-5);
    EXPECT_NEAR(output_b[i], input[i], 1e-5);
  }
}

class PoolingTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, Shape, Shape, std::string,
                     std::vector<double> > > {};
// 1) input; 2) input_shape; 3) pooling_shape; 4) pooling_type;
// 5) expected_output.

TEST_P(PoolingTestsParameterized, pooling_works_correctly) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  Shape inpshape = std::get<1>(data);
  Shape poolshape = std::get<2>(data);
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, std::get<3>(data));
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = std::get<4>(data);
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR(output[i], true_output[i], 1e-5);
  }
}

std::vector<double> basic_1d_data = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0};
Shape basic_1d_shape = {8};

std::vector<double> basic_2d_1_data = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                                       2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
Shape basic_2d_1_shape = {4, 4};

std::vector<double> basic_2d_2_data = {9.0, 8.0, 7.0, 5.0, 4.0,
                                       3.0, 2.0, 3.0, 4.0};
Shape basic_2d_2_shape = {3, 3};

std::vector<double> basic_4d_data = {
    2.0, 3.0, 1.0, 4.0,  0.0,  3.0, 7.0, 1.0, 3.0, 7.0,  0.0,  7.0,
    0.0, 8.0, 0.0, -1.0, 8.0,  1.0, 1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
    7.0, 8.0, 9.0, 10.0, 12.0, 2.0, 0.0, 9.0, 8.0, 17.0, -1.0, 120.0};
Shape basic_4d_shape = {2, 2, 3, 3};

INSTANTIATE_TEST_SUITE_P(
    pooling_tests, PoolingTestsParameterized,
    ::testing::Values(
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}),
                        std::string("average"),
                        std::vector<double>({8.0, 5.0})),
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}),
                        std::string("max"), std::vector<double>({9.0, 6.0})),
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({8}),
                        std::string("average"), std::vector<double>({5.5})),
        std::make_tuple(basic_2d_1_data, basic_2d_1_shape, Shape({2, 2}),
                        std::string("average"),
                        std::vector<double>({6.5, 4.5, 4.5, 6.5})),
        std::make_tuple(basic_2d_1_data, basic_2d_1_shape, Shape({2, 2}),
                        std::string("max"),
                        std::vector<double>({9.0, 7.0, 7.0, 9.0})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({2, 2}),
                        std::string("average"), std::vector<double>({6.5})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({2, 2}),
                        std::string("max"), std::vector<double>({9.0})),
        std::make_tuple(basic_2d_2_data, basic_2d_2_shape, Shape({3, 3}),
                        std::string("average"), std::vector<double>({5.0})),
        std::make_tuple(basic_4d_data, basic_4d_shape, Shape({2, 2}),
                        std::string("max"),
                        std::vector<double>({4.0, 8.0, 5.0, 12.0}))));

TEST(poolinglayer, new_pooling_layer_can_run_float_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F,
                            2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  std::vector<float> true_output = {6.5F, 4.5F, 4.5F, 6.5F};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, new_pooling_layer_can_run_int_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  std::vector<int> true_output = {6, 4, 4, 6};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<int>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, new_pooling_layer_can_run_int_avg_tbb) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average", it_lab_ai::kTBB);
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  std::vector<int> true_output = {6, 4, 4, 6};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<int>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, new_pooling_layer_can_run_1d_pooling_float) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  std::vector<float> true_output = {8.0F, 5.0F};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, new_pooling_layer_tbb_can_run_1d_pooling_float) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer a(poolshape, "average", it_lab_ai::kTBB);
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  std::vector<float> true_output = {8.0F, 5.0F};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, IncompatibleInput) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape),
                         make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  EXPECT_THROW(a.run(in, out), std::runtime_error);
}