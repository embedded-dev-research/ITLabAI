#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/Tensor.hpp"
#include "layers/TransposeLayer.hpp"

using namespace it_lab_ai;

TEST(TransposeLayerTest, EmptyTensor) {
  Tensor input = make_tensor<float>({}, {0});
  TransposeLayer layer;
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(TransposeLayerTest, InvalidInput) {
  Tensor input = make_tensor<float>({}, {0});
  TransposeLayer layer;
  Tensor output;
  std::vector<Tensor> in{input, input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(TransposeLayerTest, IdentityTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 1});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1}), 4.0f);
}

TEST(TransposeLayerTest, VectorTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {4});
  TransposeLayer layer({0});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({4}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({2}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({3}), 4.0f);
}

TEST(TransposeLayerTest, InvalidPermutationSize) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(TransposeLayerTest, DuplicateAxes) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 0});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(TransposeLayerTest, NegativeAxis) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, -1});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(TransposeLayerTest, LargeAxis) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({0, 2});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::invalid_argument);
}

TEST(TransposeLayerTest, 4DTensorTranspose) {
  std::vector<float> data(16);
  std::iota(data.begin(), data.end(), 1.0f);
  Tensor input = make_tensor<float>(data, {2, 2, 2, 2});
  TransposeLayer layer({3, 1, 0, 2});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2, 2, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0, 1}), 7.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 0, 1}), 8.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 0}), 9.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 1, 0}), 10.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 1, 0}), 13.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 1, 0}), 14.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 1}), 11.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 1, 1}), 12.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 1, 1}), 15.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 1, 1}), 16.0f);
}

TEST(TransposeLayerTest, MatrixTranspose) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({1, 0});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1}), 4.0f);
}

TEST(TransposeLayerTest, 3DTensor) {
  std::vector<float> data(24);
  std::iota(data.begin(), data.end(), 1.0f);
  Tensor input = make_tensor<float>(data, {2, 3, 4});
  TransposeLayer layer({2, 0, 1});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  ASSERT_EQ(out[0].get_shape(), Shape({4, 2, 3}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({3, 1, 2}), 24.0f);
}

TEST(TransposeLayerTest, IntTensor) {
  Tensor input = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  TransposeLayer layer({1, 0});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  EXPECT_EQ(out[0].get<int>({0, 0}), 1);
  EXPECT_EQ(out[0].get<int>({1, 0}), 2);
  EXPECT_EQ(out[0].get<int>({0, 1}), 3);
  EXPECT_EQ(out[0].get<int>({1, 1}), 4);
}

TEST(TransposeLayerTest, 1DDefaultPermutationIsNoOp) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4});

  TransposeLayer layer;
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  EXPECT_EQ(out[0].get_shape(), Shape({4}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({2}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({3}), 4.0f);
}

TEST(TransposeLayerTest, MultipleRunsWithDifferentRanks) {
  TransposeLayer layer({1, 0});

  {
    Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};
    layer.run(in, out);
    EXPECT_EQ(out[0].get_shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 2.0f);
  }

  {
    Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};
    layer.run(in, out);
    EXPECT_EQ(out[0].get_shape(), Shape({2, 3}));
  }

  {
    Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_THROW(layer.run(in, out), std::invalid_argument);
  }
}

TEST(TransposeLayerTest, ExplicitPermutationWithDifferentRanks) {
  TransposeLayer layer({1, 0});

  Tensor input2D = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  Tensor output2D;
  std::vector<Tensor> in{input2D};
  std::vector<Tensor> out{output2D};
  layer.run(in, out);
  EXPECT_EQ(out[0].get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 2.0f);

  Tensor input3D = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor output3D;
  std::vector<Tensor> in3{input3D};
  std::vector<Tensor> out3{output3D};
  EXPECT_THROW(layer.run(in3, out3), std::invalid_argument);
}