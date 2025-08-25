#include <gtest/gtest.h>

#include "layers/ReduceLayer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

TEST(ReduceLayer, DefaultConstructor) { ASSERT_NO_THROW(ReduceLayer layer); }

TEST(ReduceLayer, SumAllAxesKeepDims) {
  ReduceLayer layer(1);
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 10.0f);
}

TEST(ReduceLayer, SumAlongAxis0) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(0, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 4.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 6.0f);
}

TEST(ReduceLayer, SumAlongAxis1KeepDims) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({1});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 7.0f);
}

TEST(ReduceLayer, IncompatibleInput) {
  Tensor input = make_tensor<float>({1.0f, 2.0f}, {2});
  Tensor axes = make_tensor<int>({2});
  ReduceLayer layer(0, axes);

  Tensor output;

  std::vector<Tensor> in{input, input};
  std::vector<Tensor> out{output};
  ASSERT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(ReduceLayer, InvalidAxisThrows) {
  Tensor input = make_tensor<float>({1.0f, 2.0f}, {2});
  Tensor axes = make_tensor<int>({2});
  ReduceLayer layer(0, axes);

  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  ASSERT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(ReduceLayer, IntTensorSupport) {
  Tensor input = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(0, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_EQ(out[0].get<int>({0}), 4);
  EXPECT_EQ(out[0].get<int>({1}), 6);
}

TEST(ReduceLayer, 3DTensorReduction) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({2});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 2, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0}), 7.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0}), 11.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 0}), 15.0f);
}

TEST(ReduceLayer, 3DReductionAxis2) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({1});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 1, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 1}), 14.0f);
}

TEST(ReduceLayer, 3DReductionAxis10) {
  Tensor input = make_tensor<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {2, 2, 2, 2});

  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 2, 2, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0}), 1 + 9);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 1}), 2 + 10);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 0}), 3 + 11);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1, 1}), 4 + 12);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0, 0}), 5 + 13);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0, 1}), 6 + 14);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 1, 0}), 7 + 15);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 1, 1}), 8 + 16);
}

TEST(ReduceLayer, 3DFullReduction) {
  ReduceLayer layer(1);
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 1, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}

TEST(ReduceLayer, Resnet) {
  Tensor input = make_tensor<float>(
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
       10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
       19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f,
       28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f,
       37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
       46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f},
      {1, 2, 3, 3, 3});

  Tensor axes = make_tensor<int>({1});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 1, 3, 3, 3}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0, 0, 0}), 1.0f + 28.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 2, 2, 2}), 27.0f + 54.0f);
}

TEST(ReduceLayer, NegativeAxisBasic) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({-1});
  ReduceLayer layer(0, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 7.0f);
}

TEST(ReduceLayer, NegativeAxis3DTensor) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({-2});
  ReduceLayer layer(1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 1, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 1}), 14.0f);
}

TEST(ReduceLayer, ReduceMean) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor output;
  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(ReduceLayer::Operation::kMean, 1, axes);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 2.0f);
}

TEST(ReduceLayer, ReduceMeanResnet) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor output;
  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(ReduceLayer::Operation::kMean, 1, axes);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 2.0f);
}

TEST(ReduceLayer, MultAlongAxis0) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({0});
  ReduceLayer layer(ReduceLayer::Operation::kMult, 0, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1}), 8.0f);
}

TEST(ReduceLayer, MaxAlongAxis1KeepDims) {
  Tensor input = make_tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor axes = make_tensor<int>({1});
  ReduceLayer layer(ReduceLayer::Operation::kMax, 1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0}), 4.0f);
}

TEST(ReduceLayer, Min3DTensorReduction) {
  Tensor input = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  Tensor axes = make_tensor<int>({2});
  ReduceLayer layer(ReduceLayer::Operation::kMin, 1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({2, 2, 1}));
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(out[0].get<float>({1, 1, 0}), 7.0f);
}

TEST(ReduceLayer, ResnetReduceMean) {
  Tensor input = make_tensor<float>(
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
       10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
       19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f},
      {1, 1, 3, 3, 3});

  Tensor axes = make_tensor<int>({2, 3});

  ReduceLayer layer(ReduceLayer::Operation::kMean, 1, axes);
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  EXPECT_EQ(out[0].get_shape(), Shape({1, 1, 1, 1, 3}));
  EXPECT_FLOAT_EQ(
      out[0].get<float>({0, 0, 0, 0, 0}),
      (1.0f + 4.0f + 7.0f + 10.0f + 13.0f + 16.0f + 19.0f + 22.0f + 25.0f) /
          9.0f);

  EXPECT_FLOAT_EQ(
      out[0].get<float>({0, 0, 0, 0, 1}),
      (2.0f + 5.0f + 8.0f + 11.0f + 14.0f + 17.0f + 20.0f + 23.0f + 26.0f) /
          9.0f);

  EXPECT_FLOAT_EQ(
      out[0].get<float>({0, 0, 0, 0, 2}),
      (3.0f + 6.0f + 9.0f + 12.0f + 15.0f + 18.0f + 21.0f + 24.0f + 27.0f) /
          9.0f);
}

}  // namespace it_lab_ai