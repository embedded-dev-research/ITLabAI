#include <vector>

#include "gtest/gtest.h"
#include "layers/SplitLayer.hpp"
#include "layers/Tensor.hpp"

using namespace it_lab_ai;

TEST(SplitLayerTests, SplitEqualParts1D) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {6});
  SplitLayer splitter(0, 3);

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2}));
  EXPECT_EQ(outputs[2].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({0}), 1.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({0}), 3.0f);
  EXPECT_FLOAT_EQ(outputs[2].get<float>({0}), 5.0f);
}

TEST(SplitLayerTests, SplitVariableParts1D) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {6});
  SplitLayer splitter(0, {2, 4});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({4}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({1}), 2.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({3}), 6.0f);
}

TEST(SplitLayerTests, Split2DAlongAxis0) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  SplitLayer splitter(0, {1, 1});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({1, 3}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({1, 3}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({0, 0}), 4.0f);
}

TEST(SplitLayerTests, Split2DAlongAxis1) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  SplitLayer splitter(1, {1, 2});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2, 1}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({0, 1}), 3.0f);
}

TEST(SplitLayerTests, Split3DEqualParts) {
  std::vector<float> data(2 * 3 * 4);
  std::iota(data.begin(), data.end(), 0.0f);
  Tensor input = make_tensor<float>(data, {2, 3, 4});

  SplitLayer splitter(1, 3);

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2, 1, 4}));
  EXPECT_EQ(outputs[1].get<float>({1, 0, 3}), 19.0f);
}

TEST(SplitLayerTests, Split4DVariableParts) {
  std::vector<float> data(1 * 3 * 2 * 4);
  std::iota(data.begin(), data.end(), 0.0f);
  Tensor input = make_tensor<float>(data, {1, 3, 2, 4});

  SplitLayer splitter(2, {1, 1});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({1, 3, 1, 4}));
  EXPECT_EQ(outputs[1].get<float>({0, 2, 0, 3}), 23.0f);
}

TEST(SplitLayerTests, SplitNegativeAxis) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  SplitLayer splitter(-1, {1, 2});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2, 1}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2, 2}));
}

TEST(SplitLayerTests, InvalidSplitSizes) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {4});

  SplitLayer splitter(0, {1, 2});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};

  EXPECT_THROW(splitter.run(in, outputs), std::runtime_error);
}

TEST(SplitLayerTests, InvalidInput) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {4});

  SplitLayer splitter(0, {1, 2});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input, input};

  EXPECT_THROW(splitter.run(in, outputs), std::runtime_error);
}

TEST(SplitLayerTests, EmptyInputTensor) {
  Tensor input = make_tensor<float>({}, {0});

  SplitLayer splitter(0, {});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};

  EXPECT_THROW(splitter.run(in, outputs), std::runtime_error);
}

TEST(SplitLayerTests, Split192IntoTwo96) {
  std::vector<float> input_data(1 * 192 * 56 * 56);
  std::iota(input_data.begin(), input_data.end(), 0.0f);
  Tensor input = make_tensor<float>(input_data, {1, 192, 56, 56});

  SplitLayer splitter(1, {96, 96});
  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({1, 96, 56, 56}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({1, 96, 56, 56}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({0, 0, 0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({0, 0, 0, 0}), 96 * 56 * 56);
}

TEST(SplitLayerTests, UnevenSplitWithRemainder) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5}, {5});
  SplitLayer splitter(0, 3);

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2}));
  EXPECT_EQ(outputs[2].get_shape(), Shape({1}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({1}), 2.0f);
  EXPECT_FLOAT_EQ(outputs[1].get<float>({1}), 4.0f);
  EXPECT_FLOAT_EQ(outputs[2].get<float>({0}), 5.0f);
}

TEST(SplitLayerTests, NumOutputsGreaterThanAxisSize) {
  Tensor input = make_tensor<float>({1, 2, 3}, {3});
  SplitLayer splitter(0, 5);

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};

  EXPECT_THROW(splitter.run(in, outputs), std::runtime_error);
}

TEST(SplitLayerTests, IntegerDataType) {
  Tensor input = make_tensor<int>({1, 2, 3, 4, 5, 6}, {2, 3});
  SplitLayer splitter(1, {1, 2});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2, 1}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2, 2}));
  EXPECT_EQ(outputs[0].get<int>({1, 0}), 4);
  EXPECT_EQ(outputs[1].get<int>({0, 1}), 3);
}

TEST(SplitLayerTests, NegativeAxis2D) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});
  SplitLayer splitter(-2, {1, 1});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({1, 2}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({1, 2}));
}

TEST(SplitLayerTests, NegativeAxis3D) {
  std::vector<float> data(2 * 3 * 4);
  std::iota(data.begin(), data.end(), 1.0f);
  Tensor input = make_tensor<float>(data, {2, 3, 4});

  SplitLayer splitter(-1, {1, 3});

  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};
  splitter.run(in, outputs);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].get_shape(), Shape({2, 3, 1}));
  EXPECT_EQ(outputs[1].get_shape(), Shape({2, 3, 3}));
  EXPECT_FLOAT_EQ(outputs[0].get<float>({1, 2, 0}), 21.0f);
}

TEST(SplitLayerTests, LargeAxisValue) {
  Tensor input = make_tensor<float>({1, 2, 3, 4}, {2, 2});

  SplitLayer splitter(10, {1, 1});
  std::vector<Tensor> outputs;
  std::vector<Tensor> in{input};

  EXPECT_THROW(splitter.run(in, outputs), std::runtime_error);
}
