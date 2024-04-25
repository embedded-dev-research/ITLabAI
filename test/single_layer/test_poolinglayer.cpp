#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

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
  EXPECT_EQ(output_a, input);
  EXPECT_EQ(output_b, input);
}

TEST(poolinglayer, 1d_pooling_avg_test) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {8.0, 5.0, 2.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 1d_pooling_max_test) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 6.0, 3.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 1d_bigger_pooling_test) {
  Shape inpshape = {8};
  Shape poolshape = {9};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {5.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_avg_test1) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0,
                             4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {6.5, 4.5, 4.5, 6.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_avg_test2) {
  Shape inpshape = {3, 3};
  Shape poolshape = {2, 2};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {6.5, 5.0, 2.5, 4.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_max_test1) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0,
                             4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 7.0, 7.0, 9.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_max_test2) {
  Shape inpshape = {3, 3};
  Shape poolshape = {2, 2};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 7.0, 3.0, 4.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_bigger_pooling_test) {
  Shape inpshape = {3, 3};
  Shape poolshape = {4, 4};
  PoolingLayerImpl<double> a =
      PoolingLayerImpl<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {5.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, new_pooling_layer_can_run_float_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F,
                            2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<float> true_output = {6.5F, 4.5F, 4.5F, 6.5F};
  EXPECT_EQ(*output.as<float>(), true_output);
}

TEST(poolinglayer, new_pooling_layer_can_run_int_avg) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer a(poolshape, "average");
  std::vector<int> input({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<int> true_output = {6, 4, 4, 6};
  EXPECT_EQ(*output.as<int>(), true_output);
}

TEST(poolinglayer, new_pooling_layer_can_run_1d_pooling_float) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer a(poolshape, "average");
  std::vector<float> input({9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  Tensor output = make_tensor<float>({0});
  a.run(make_tensor(input, inpshape), output);
  std::vector<float> true_output = {8.0F, 5.0F, 2.5F};
  EXPECT_EQ(*output.as<float>(), true_output);
}
