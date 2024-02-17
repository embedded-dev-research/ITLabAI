#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

TEST(poolinglayer, empty_inputs1) {
  Shape inpshape = 0;
  Shape poolshape = 0;
  ASSERT_ANY_THROW(PoolingLayer<double> a =
                       PoolingLayer<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, empty_inputs2) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input;
  ASSERT_ANY_THROW(std::vector<double> output = a.run(input));
}

TEST(poolinglayer, throws_when_bigger_pooling_dims) {
  Shape inpshape = {8};
  Shape poolshape = {8, 8};
  ASSERT_ANY_THROW(PoolingLayer<double> a =
                       PoolingLayer<double>(inpshape, poolshape, "average"));
}

TEST(poolinglayer, equivalent_output_when_pool_size_1) {
  Shape inpshape = {8};
  Shape poolshape = {1};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  PoolingLayer<double> b = PoolingLayer<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output_a = a.run(input);
  std::vector<double> output_b = b.run(input);
  EXPECT_EQ(output_a, input);
  EXPECT_EQ(output_b, input);
}

TEST(poolinglayer, 1d_pooling_avg_test) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {8.0, 5.0, 2.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 1d_pooling_max_test) {
  Shape inpshape = {8};
  Shape poolshape = {3};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 6.0, 3.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 1d_bigger_pooling_test) {
  Shape inpshape = {8};
  Shape poolshape = {9};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {5.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_avg_test1) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0,
                             4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {6.5, 4.5, 4.5, 6.5};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_avg_test2) {
  Shape inpshape = {3, 3};
  Shape poolshape = {2, 2};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {6.5, 5.0, 2.5, 4.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_max_test1) {
  Shape inpshape = {4, 4};
  Shape poolshape = {2, 2};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0,
                             4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 7.0, 7.0, 9.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_pooling_max_test2) {
  Shape inpshape = {3, 3};
  Shape poolshape = {2, 2};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "max");
  std::vector<double> input({9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {9.0, 7.0, 3.0, 4.0};
  EXPECT_EQ(output, true_output);
}

TEST(poolinglayer, 2d_bigger_pooling_test) {
  Shape inpshape = {3, 3};
  Shape poolshape = {4, 4};
  PoolingLayer<double> a = PoolingLayer<double>(inpshape, poolshape, "average");
  std::vector<double> input({9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = {5.0};
  EXPECT_EQ(output, true_output);
}
