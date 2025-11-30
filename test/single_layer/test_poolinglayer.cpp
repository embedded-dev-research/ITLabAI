#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PoolingTestsParameterized);

TEST(poolinglayer, empty_inputs1) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  EXPECT_NO_THROW(PoolingLayerImpl<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, empty_inputs2) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input;
  ASSERT_ANY_THROW(std::vector<double> output = a.run(input));
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
  PoolingLayerImplTBB<double> a = PoolingLayerImplTBB<double>(
      inpshape, poolshape, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "average");
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
  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, {1}, {0, 0, 0, 0}, {1, 1}, false, "average");
  PoolingLayerImpl<double> b = PoolingLayerImpl<double>(
      inpshape, poolshape, {1}, {0, 0, 0, 0}, {1, 1}, false, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output_a = a.run(input);
  std::vector<double> output_b = b.run(input);

  EXPECT_EQ(output_a.size(), input.size());
  EXPECT_EQ(output_b.size(), input.size());

  for (size_t i = 0; i < output_a.size(); i++) {
    EXPECT_NEAR(output_a[i], input[i], 1e-5);
    EXPECT_NEAR(output_b[i], input[i], 1e-5);
  }
}

TEST(poolinglayer, different_strides) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, {3}, {0, 0, 0, 0}, {1, 1}, false, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  EXPECT_NEAR(output[0], 8.0, 1e-5);
  EXPECT_NEAR(output[1], 5.0, 1e-5);
}

TEST(poolinglayer, with_padding) {
  Shape inpshape = {4};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, {1}, {1, 1, 0, 0}, {1, 1}, false, "average");
  std::vector<double> input({1.0, 2.0, 3.0, 4.0});
  std::vector<double> output = a.run(input);
  EXPECT_NEAR(output[0], 1.5, 1e-5);
  EXPECT_NEAR(output[1], 2.0, 1e-5);
  EXPECT_NEAR(output[2], 3.0, 1e-5);
  EXPECT_NEAR(output[3], 3.5, 1e-5);
}

TEST(poolinglayer, with_dilation) {
  Shape inpshape = {6};
  Shape poolshape = {2};
  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, {1}, {0, 0, 0, 0}, {2, 1}, false, "max");
  std::vector<double> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  std::vector<double> output = a.run(input);
  EXPECT_NEAR(output[0], 3.0, 1e-5);
  EXPECT_NEAR(output[1], 4.0, 1e-5);
  EXPECT_NEAR(output[2], 5.0, 1e-5);
  EXPECT_NEAR(output[3], 6.0, 1e-5);
}

TEST(poolinglayer, ceil_mode_vs_floor_mode) {
  Shape inpshape = {5};
  Shape poolshape = {3};

  PoolingLayerImpl<double> floor_mode = PoolingLayerImpl<double>(
      inpshape, poolshape, {2}, {0, 0, 0, 0}, {1, 1}, false, "average");

  PoolingLayerImpl<double> ceil_mode = PoolingLayerImpl<double>(
      inpshape, poolshape, {2}, {0, 0, 0, 0}, {1, 1}, true, "average");

  std::vector<double> input({1.0, 2.0, 3.0, 4.0, 5.0});

  std::vector<double> floor_output = floor_mode.run(input);
  std::vector<double> ceil_output = ceil_mode.run(input);

  EXPECT_EQ(floor_output.size(), 2);
  EXPECT_EQ(ceil_output.size(), 2);
}

TEST(poolinglayer, 2d_with_complex_parameters) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, {2, 2}, {1, 1, 1, 1}, {1, 1}, false, "max");

  std::vector<double> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                             11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

  std::vector<double> output = a.run(input);
  EXPECT_EQ(output.size(), 9);
}

class PoolingTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, Shape, Shape, Shape, Shape, Shape,
                     bool, std::string, std::vector<double>>> {};
// 1) input; 2) input_shape; 3) pooling_shape; 4) strides; 5) pads; 6)
// dilations; 7) ceil_mode; 8) pooling_type; 9) expected_output.

TEST_P(PoolingTestsParameterized, pooling_works_correctly_with_parameters) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  Shape inpshape = std::get<1>(data);
  Shape poolshape = std::get<2>(data);
  Shape strides = std::get<3>(data);
  Shape pads = std::get<4>(data);
  Shape dilations = std::get<5>(data);
  bool ceil_mode = std::get<6>(data);
  std::string pooling_type = std::get<7>(data);

  PoolingLayerImpl<double> a = PoolingLayerImpl<double>(
      inpshape, poolshape, strides, pads, dilations, ceil_mode, pooling_type);

  std::vector<double> output = a.run(input);
  std::vector<double> true_output = std::get<8>(data);

  ASSERT_EQ(output.size(), true_output.size());
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

INSTANTIATE_TEST_SUITE_P(
    pooling_tests, PoolingTestsParameterized,
    ::testing::Values(
        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}), Shape({2}),
                        Shape({0, 0, 0, 0}), Shape({1, 1}), false, "average",
                        std::vector<double>({8.0, 6.0, 4.0})),

        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}), Shape({2}),
                        Shape({0, 0, 0, 0}), Shape({1, 1}), false, "max",
                        std::vector<double>({9.0, 7.0, 5.0})),

        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}), Shape({3}),
                        Shape({0, 0, 0, 0}), Shape({1, 1}), false, "average",
                        std::vector<double>({8.0, 5.0})),

        std::make_tuple(basic_1d_data, basic_1d_shape, Shape({3}), Shape({1}),
                        Shape({1, 1, 0, 0}), Shape({1, 1}), false, "average",
                        std::vector<double>({8.5, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                                             2.5})),

        std::make_tuple(basic_2d_1_data, basic_2d_1_shape, Shape({2, 2}),
                        Shape({1, 1}), Shape({0, 0, 0, 0}), Shape({1, 1}),
                        false, "average",
                        std::vector<double>({6.5, 5.5, 4.5, 3.5, 3.5, 3.5, 4.5,
                                             5.5, 6.5}))));

TEST(poolinglayer, new_pooling_layer_can_run_float_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};

  PoolingLayer a(poolshape, "average");

  PoolingLayerImpl<float> impl(inpshape, poolshape, "average");

  Shape output_shape = impl.get_output_shape();
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F,
                            2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  std::vector<float> zeros(output_shape.count(), 0.0f);
  Tensor output = make_tensor(zeros, output_shape);

  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};

  a.run(in, out);

  std::vector<float> true_output = {6.5F, 4.5F, 4.5F, 6.5F};
  for (size_t i = 0; i < true_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], true_output[i], 1e-5);
  }
}

TEST(poolinglayer, new_pooling_layer_with_parameters) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, {1, 1}, {1, 1, 1, 1}, {1, 1}, false, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F,
                            2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor output = make_tensor<float>({0});
  std::vector<Tensor> in{make_tensor(input, inpshape)};
  std::vector<Tensor> out{output};
  a.run(in, out);
  EXPECT_EQ(out[0].get_shape().count(), 25);
}

TEST(poolinglayer, new_pooling_layer_can_run_int_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "average");
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});

  PoolingLayerImpl<int> impl(inpshape, poolshape, {2, 2}, {0, 0, 0, 0}, {1, 1},
                             false, "average");
  Shape output_shape = impl.get_output_shape();

  std::vector<int> zeros(output_shape.count(), 0);
  Tensor output = make_tensor(zeros, output_shape);

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
  PoolingLayer a(poolshape, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "average",
                 it_lab_ai::kTBB);
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});

  PoolingLayerImplTBB<int> impl(inpshape, poolshape, {2, 2}, {0, 0, 0, 0},
                                {1, 1}, false, "average");
  Shape output_shape = impl.get_output_shape();

  std::vector<int> zeros(output_shape.count(), 0);
  Tensor output = make_tensor(zeros, output_shape);

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
  std::vector<float> true_output = {8.0F, 6.0F, 4.0F};
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
  std::vector<float> true_output = {8.0F, 6.0F, 4.0F};
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

TEST(poolinglayer, maxpool_onnx_example) {
  Shape input_shape = {1, 64, 112, 112};
  Shape poolshape = {3, 3};
  Shape strides = {2, 2};
  Shape pads = {0, 0, 0, 0};
  Shape dilations = {1, 1};
  bool ceil_mode = true;
  std::string pooling_type = "max";

  PoolingLayerImpl<float> impl(input_shape, poolshape, strides, pads, dilations,
                               ceil_mode, pooling_type);

  Shape expected_output_shape = {1, 64, 56, 56};
  EXPECT_EQ(impl.get_output_shape(), expected_output_shape);

  std::vector<float> input(input_shape.count());
  for (size_t i = 0; i < input.size(); i++) {
    input[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
  }

  std::vector<float> output = impl.run(input);

  EXPECT_EQ(output.size(), expected_output_shape.count());

  for (float val : output) {
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 10.0f);
  }

  float first_window_max = 0.0f;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 3; l++) {
      size_t pos = k * 112 + l;
      if (pos < input.size()) {
        first_window_max = std::max(first_window_max, input[pos]);
      }
    }
  }

  EXPECT_NEAR(output[0], first_window_max, 1e-5);
}

TEST(poolinglayer, maxpool_onnx_with_pooling_layer) {
  Shape input_shape = {1, 64, 112, 112};
  Shape poolshape = {3, 3};
  Shape strides = {2, 2};
  Shape pads = {0, 0, 0, 0};
  Shape dilations = {1, 1};
  bool ceil_mode = true;

  PoolingLayer layer(poolshape, strides, pads, dilations, ceil_mode, "max");

  std::vector<float> input(input_shape.count());
  for (size_t i = 0; i < input.size(); i++) {
    input[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
  }

  Tensor input_tensor = make_tensor(input, input_shape);

  PoolingLayerImpl<float> impl(input_shape, poolshape, strides, pads, dilations,
                               ceil_mode, "max");
  Shape output_shape = impl.get_output_shape();
  std::vector<float> zeros(output_shape.count(), 0.0f);
  Tensor output_tensor = make_tensor(zeros, output_shape);

  std::vector<Tensor> inputs{input_tensor};
  std::vector<Tensor> outputs{output_tensor};

  layer.run(inputs, outputs);

  EXPECT_EQ(outputs[0].get_shape(), output_shape);

  auto output_data = *outputs[0].as<float>();
  for (float val : output_data) {
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 10.0f);
  }
}
