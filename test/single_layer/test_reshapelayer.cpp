#include <vector>

#include "gtest/gtest.h"
#include "layers/ReshapeLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(ReshapeLayerTest, BasicReshape2DTo3D) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor input = make_tensor(data, {2, 6});
  Tensor output;
  ReshapeLayer layer(false, {2, 3, 2});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 3, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 2, 1}), 12.0f);
}

TEST(ReshapeLayerTest, BasicReshape3DTo2D) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor input = make_tensor(data, {2, 2, 3});
  Tensor output;
  ReshapeLayer layer(false, {4, 3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({4, 3}));
  EXPECT_EQ(out[0].get<int>({0, 0}), 1);
  EXPECT_EQ(out[0].get<int>({0, 1}), 2);
  EXPECT_EQ(out[0].get<int>({3, 2}), 12);
}

TEST(ReshapeLayerTest, NegativeDimensionInference) {
  std::vector<float> data(12, 1.0f);
  Tensor input = make_tensor(data, {2, 6});
  Tensor output;
  ReshapeLayer layer(false, {2, -1, 2});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 3, 2}));
}

TEST(ReshapeLayerTest, ZeroDimensionCopy) {
  std::vector<int> data(24, 5);
  Tensor input = make_tensor(data, {2, 3, 4});
  Tensor output;
  ReshapeLayer layer(true, {2, 0, 4});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 3, 4}));
}

TEST(ReshapeLayerTest, FlattenTo1D) {
  std::vector<float> data;
  for (int i = 0; i < 24; ++i) data.push_back(static_cast<float>(i));

  Tensor input = make_tensor(data, {2, 3, 4});
  Tensor output;
  ReshapeLayer layer(false, {-1});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({24}));
  for (size_t i = 0; i < 24; ++i) {
    EXPECT_FLOAT_EQ(out[0].get<float>({i}), static_cast<float>(i));
  }
}

TEST(ReshapeLayerTest, TotalElementsMismatchError) {
  std::vector<float> data(6, 1.0f);
  Tensor input = make_tensor(data, {6});
  Tensor output;
  ReshapeLayer layer(false, {2, 4});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(ReshapeLayerTest, MultipleNegativeOnesError) {
  std::vector<float> data(6, 1.0f);
  Tensor input = make_tensor(data, {6});
  Tensor output;
  ReshapeLayer layer(false, {2, -1, -1});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(ReshapeLayerTest, ZeroDimensionWithoutAllowZero) {
  std::vector<float> data(6, 1.0f);
  Tensor input = make_tensor(data, {6});
  Tensor output;
  ReshapeLayer layer(false, {2, 0, 3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(ReshapeLayerTest, NegativeDimensionIndexError) {
  std::vector<float> data(6, 1.0f);
  Tensor input = make_tensor(data, {6});
  Tensor output;
  ReshapeLayer layer(false, {2, -2, 3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::length_error);
}

TEST(ReshapeLayerTest, ZeroDimensionIndexOutOfRange) {
  std::vector<float> data(6, 1.0f);
  Tensor input = make_tensor(data, {2, 3});
  Tensor output;
  ReshapeLayer layer(true, {2, 0, 3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(ReshapeLayerTest, EmptyOutputShape) {
  std::vector<int> data = {1, 2, 3};
  Tensor input = make_tensor(data, {3});
  Tensor output;

  ReshapeLayer layer(false, {3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  ASSERT_EQ(out[0].get_shape(), Shape({3}));
}

TEST(ReshapeLayerTest, ComplexReshapeWithNegativeOne) {
  std::vector<int> data(2 * 3 * 4 * 5, 7);
  Tensor input = make_tensor(data, {2, 3, 4, 5});
  Tensor output;
  ReshapeLayer layer(false, {2, -1, 5});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 12, 5}));
  EXPECT_EQ(out[0].get<int>({0, 0, 0}), 7);
  EXPECT_EQ(out[0].get<int>({1, 11, 4}), 7);
}

TEST(ReshapeLayerTest, AllowZeroFalseWithValidShape) {
  std::vector<float> data(1 * 6 * 64 * 49, 1.0f);
  Tensor input = make_tensor(data, {1, 6, 64, 49});
  Tensor output;

  ReshapeLayer layer(false, {1, 384, 7, 7});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  ASSERT_EQ(out[0].get_shape(), Shape({1, 384, 7, 7}));
}

TEST(ReshapeLayerTest, BatchReshapeSingleToBatch) {
  std::vector<float> data(2 * 768 * 7 * 7, 1.5f);
  Tensor input = make_tensor(data, {2, 768, 7, 7});
  Tensor output;
  ReshapeLayer layer(false, {1, 6, 128, 49});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  ASSERT_EQ(out[0].get_shape(), Shape({2, 6, 128, 49}));

  EXPECT_EQ(out[0].get<float>({0, 0, 0, 0}), 1.5f);
  EXPECT_EQ(out[0].get<float>({1, 5, 127, 48}), 1.5f);
}

TEST(ReshapeLayerTest, BatchReshapeWithNegativeOneAndBatch) {
  std::vector<float> data(4 * 3 * 10 * 10, 3.14f);
  Tensor input = make_tensor(data, {4, 3, 10, 10});
  Tensor output;

  ReshapeLayer layer(false, {1, -1, 5});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  ASSERT_EQ(out[0].get_shape(), Shape({4, 60, 5}));
  EXPECT_EQ(out[0].get<float>({0, 0, 0}), 3.14f);
  EXPECT_EQ(out[0].get<float>({3, 59, 4}), 3.14f);
}

TEST(ReshapeLayerTest, BatchReshapeWithZeroDimAndBatch) {
  std::vector<int> data(2 * 6 * 8 * 8, 99);
  Tensor input = make_tensor(data, {2, 6, 8, 8});
  Tensor output;

  ReshapeLayer layer(false, {1, 0, 16, 4});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  ASSERT_EQ(out[0].get_shape(), Shape({2, 6, 16, 4}));
  EXPECT_EQ(out[0].get<int>({0, 0, 0, 0}), 99);
  EXPECT_EQ(out[0].get<int>({1, 5, 15, 3}), 99);
}

TEST(ReshapeLayerTest, BatchReshapeComplexYOLOLike) {
  std::vector<float> data(2 * 768 * 7 * 7, 0.5f);
  Tensor input = make_tensor(data, {2, 768, 7, 7});
  Tensor output;

  ReshapeLayer layer(false, {1, 6, 128, 49});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  ASSERT_EQ(out[0].get_shape(), Shape({2, 6, 128, 49}));

  size_t total_elements = 1;
  for (size_t i = 0; i < out[0].get_shape().dims(); ++i) {
    total_elements *= out[0].get_shape()[i];
  }
  EXPECT_EQ(total_elements, 2 * 768 * 7 * 7);

  EXPECT_EQ(out[0].get<float>({0, 0, 0, 0}), 0.5f);
  EXPECT_EQ(out[0].get<float>({1, 5, 127, 48}), 0.5f);
}

TEST(ReshapeLayerTest, BatchReshapeIncompatibleElements) {
  std::vector<int> data(2 * 100, 1);
  Tensor input = make_tensor(data, {2, 100});
  Tensor output;
  ReshapeLayer layer(false, {1, 3, 3, 3});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(ReshapeLayerTest, AllowZeroTrueCopiesInputDims) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor input = make_tensor(data, {3, 4});
  Tensor output;
  ReshapeLayer layer(true, {3, 0, 1});
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({3, 4, 1}));
}

TEST(ReshapeLayerTest, ProductValidationWithNegativeOne) {
  std::vector<int> data(24, 1);
  Tensor input = make_tensor(data, {2, 3, 4});
  Tensor output;

  ReshapeLayer layer(false, {2, -1, 2});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  size_t input_product = input.get_shape().count();
  size_t output_product = out[0].get_shape().count();
  EXPECT_EQ(input_product, output_product);
  ASSERT_EQ(out[0].get_shape(), Shape({2, 6, 2}));
}

TEST(ReshapeLayerTest, AllowZeroWithNegativeOne) {
  std::vector<float> data(60, 1.0f);
  Tensor input = make_tensor(data, {3, 4, 5});
  Tensor output;

  ReshapeLayer layer(true, {3, 0, -1});

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  size_t input_product = input.get_shape().count();
  size_t output_product = out[0].get_shape().count();
  EXPECT_EQ(input_product, output_product);
  EXPECT_EQ(out[0].get_shape(), Shape({3, 4, 5}));
}