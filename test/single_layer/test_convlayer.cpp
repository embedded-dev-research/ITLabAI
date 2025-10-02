#include <gtest/gtest.h>

#include "layers/ConvLayer.hpp"

using namespace it_lab_ai;

TEST(ConvolutionalLayerTest, IncompatibleInput) {
  int step = 2;
  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<float> vec = {1, 2, 3, 4};
  Tensor input1 = make_tensor<float>(vec, {4});
  Tensor input2 = make_tensor<float>(vec, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};
  EXPECT_THROW(layer.run(in, output), std::runtime_error);
}

TEST(ConvolutionalLayerTest, FStep2) {
  std::vector<float> image;
  image.reserve(75);
  for (int i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  int step = 2;
  std::vector<float> kernelvec;
  kernelvec.reserve(3 * 3 * 3 * 3);
  for (int i = 0; i < 81; ++i) {
    kernelvec.push_back((i % 9) % 2 == 0 ? 1.0f : 0.0f);
  }
  Shape sh2({3, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  size_t out_height = (5 + 2 * 0 - 1 * (3 - 1) - 1) / 2 + 1;
  size_t out_width = (5 + 2 * 0 - 1 * (3 - 1) - 1) / 2 + 1;
  size_t expected_size = 1 * 3 * out_height * out_width;
  std::vector<float> expected_output(expected_size, 15.0f);
  Shape output_shape({1, 3, out_height, out_width});
  std::vector<float> output_vec(expected_size, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, FStep1) {
  std::vector<float> image;
  image.reserve(75);
  for (int i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  int step = 1;
  std::vector<float> kernelvec;
  kernelvec.reserve(3 * 3 * 3 * 3);
  for (int i = 0; i < 81; ++i) {
    kernelvec.push_back((i % 9) % 2 == 0 ? 1.0f : 0.0f);
  }
  Shape sh2({3, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  size_t out_height = (5 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (5 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t expected_size = 1 * 3 * out_height * out_width;
  std::vector<float> expected_output(expected_size, 15.0f);
  Shape output_shape({1, 3, out_height, out_width});
  std::vector<float> output_vec(expected_size, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, IntStep2) {
  std::vector<int> image;
  image.reserve(75);
  for (int i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 2;
  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> expected_output(12, 5);
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, IntStep1) {
  std::vector<int> image;
  image.reserve(75);
  for (int i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 1;
  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> expected_output(27, 5);
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, FloatWithBias) {
  std::vector<float> image(75, 1.0f);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);
  std::vector<float> kernelvec;
  kernelvec.reserve(3 * 3 * 3 * 3);
  for (int i = 0; i < 81; ++i) {
    kernelvec.push_back((i % 9) % 2 == 0 ? 1.0f : 0.0f);
  }
  Shape kernel_shape({3, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  std::vector<float> biasvec = {0.5f, 0.5f, 0.5f};
  Tensor bias = make_tensor(biasvec, Shape({3}));
  size_t out_height = 3;
  size_t out_width = 3;
  size_t expected_size = 1 * 3 * out_height * out_width;
  Shape output_shape({1, 3, out_height, out_width});
  std::vector<float> output_vec(expected_size, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);
  std::vector<float> expected_output(expected_size, 15.5f);
  ConvolutionalLayer layer(1, 0, 1, kernel, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, InvalidInputShapeDims) {
  std::vector<float> image(15, 1.0f);
  Shape invalid_shape({1, 3, 5});
  Tensor input = make_tensor(image, invalid_shape);

  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape({1, 3, 3, 3});
  std::vector<float> output_vec(27, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::out_of_range);
}
TEST(ConvImplTest, RunReturnsInput) {
  std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
  ConvImpl<float> conv(1, 0, 1, 2, 2, 1, 4, {0.0});

  std::vector<float> output = conv.run(input);

  ASSERT_EQ(output, input);
}
TEST(ConvolutionalLayerTest, Conv4DKern) {
  std::vector<float> image;
  image.reserve(75);
  for (int i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  int step = 1;
  std::vector<float> kernelvec;
  kernelvec.reserve(54);
  for (int i = 0; i < 54; ++i) {
    kernelvec.push_back(1);
  }
  Shape sh2({2, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  size_t out_height = (5 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (5 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t expected_size = 1 * 2 * out_height * out_width;
  std::vector<float> expected_output(expected_size, 9);
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(expected_size, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);
  ConvolutionalLayer layer(step, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}
TEST(ConvolutionalLayerTest, Conv4DKern_int) {
  std::vector<int> image;
  image.reserve(784);
  for (int i = 0; i < 784; ++i) {
    image.push_back(1);
  }
  Shape sh1({1, 1, 28, 28});
  Tensor input = make_tensor(image, sh1);

  int step = 1;
  std::vector<int> kernelvec;
  kernelvec.reserve(400);
  for (int i = 0; i < 400; ++i) {
    kernelvec.push_back(1);
  }
  Shape sh2({16, 1, 5, 5});
  Tensor kernel = make_tensor(kernelvec, sh2);
  size_t out_height = (28 + 2 * 0 - 1 * (5 - 1) - 1) / 1 + 1;
  size_t out_width = (28 + 2 * 0 - 1 * (5 - 1) - 1) / 1 + 1;
  size_t expected_size = 1 * 16 * out_height * out_width;
  std::vector<int> expected_output(expected_size, 25);
  Shape output_shape({1, 16, out_height, out_width});
  std::vector<int> output_vec(expected_size, 0);
  Tensor output = make_tensor(output_vec, output_shape);
  ConvolutionalLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, Conv4DKern_int_36) {
  std::vector<int> image;
  image.reserve(16 * 784);
  for (int i = 0; i < 16 * 784; ++i) {
    image.push_back(1);
  }
  Shape sh1({1, 16, 28, 28});
  Tensor input = make_tensor(image, sh1);
  int step = 1;
  std::vector<int> kernelvec;
  kernelvec.reserve(5 * 5 * 16 * 36);
  for (int i = 0; i < 5 * 5 * 16 * 36; ++i) {
    kernelvec.push_back(1);
  }
  Shape sh2({36, 16, 5, 5});
  Tensor kernel = make_tensor(kernelvec, sh2);
  size_t pads = (kernel.get_shape()[2] - 1) / 2;
  size_t out_height = (28 + 2 * pads - 1 * (5 - 1) - 1) / 1 + 1;
  size_t out_width = (28 + 2 * pads - 1 * (5 - 1) - 1) / 1 + 1;
  size_t expected_size = 1 * 36 * out_height * out_width;
  std::vector<int> expected_output(expected_size, 5 * 5 * 16);
  Shape output_shape({1, 36, out_height, out_width});
  std::vector<int> output_vec(expected_size, 0);
  Tensor output = make_tensor(output_vec, output_shape);
  ConvolutionalLayer layer(step, pads, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DFloatBasic) {
  std::vector<float> image(36, 1.0f);
  Shape input_shape({1, 4, 3, 3});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(36, 1.0f);
  Shape kernel_shape({4, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.1f, 0.2f, 0.3f, 0.4f};
  Tensor bias = make_tensor(biasvec, Shape({4}));

  size_t out_height = (3 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (3 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 4, out_height, out_width});
  std::vector<float> output_vec(36, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  DepthwiseConv4D<float>(input, kernel, bias, output, 1, 1, 1);

  std::vector<float> result = *output.as<float>();

  float corner_value = 4.0f + 0.1f;
  ASSERT_NEAR(result[0], corner_value, 1e-5f);

  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_GT(result[i], 0.0f);
  }
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DIntBasic) {
  std::vector<int> image = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 2, 2, 2, 2};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> biasvec = {10, 20};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<int> output_vec(2, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  DepthwiseConv4D<int>(input, kernel, bias, output, 1, 0, 1);

  std::vector<int> result = *output.as<int>();

  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 20);
  ASSERT_EQ(result[1], 72);
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DNoBias) {
  std::vector<int> image(48, 3);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec(12, 2);
  Shape kernel_shape({3, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (4 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1;
  Shape output_shape({1, 3, out_height, out_width});
  std::vector<int> output_vec(12, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  DepthwiseConv4D<int>(input, kernel, Tensor(), output, 2, 0, 1);

  std::vector<int> result = *output.as<int>();

  ASSERT_EQ(result.size(), 12);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 24);
  }
}

TEST(ConvolutionalLayerTest, Conv4DSTLFloatWithGroups) {
  std::vector<float> image(64, 1.0f);
  Shape input_shape({1, 4, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(72, 1.0f);
  Shape kernel_shape({4, 2, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 4, out_height, out_width});
  std::vector<float> output_vec(16, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4DSTL<float>(input, kernel, Tensor(), output, 1, 0, 2, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 16);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], 18.0f, 1e-5f);
  }
}

TEST(ConvolutionalLayerTest, Conv4DSTLFloatComplex) {
  std::vector<float> image = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f,
                              1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f,
                              2.0f, 3.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 5.0f,
                              2.0f, 3.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 5.0f};
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {
      1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f,  0.0f,  -1.0f,
      1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f,  0.0f,  -1.0f,
      1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f,  -1.0f, -1.0f, -1.0f,
      1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f,  -1.0f, -1.0f, -1.0f};
  Shape kernel_shape({2, 2, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.5f, 1.0f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4DSTL<float>(input, kernel, bias, output, 1, 0, 1, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 8);
}

TEST(ConvolutionalLayerTest, DepthwiseIntegration) {
  std::vector<float> image(32, 1.0f);
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(18, 1.0f);
  Shape kernel_shape({2, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (4 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(32, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 1, 1, kernel, Tensor(), kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  ASSERT_EQ(result.size(), 32);
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DWithPadding) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 1.0f, 1.0f, 1.0f};
  Shape kernel_shape({1, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (2 + 2 * 1 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 1 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 1, out_height, out_width});
  std::vector<float> output_vec(
      output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3],
      0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  DepthwiseConv4D<float>(input, kernel, Tensor(), output, 1, 1, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 9);
}

TEST(ConvolutionalLayerTest, Conv4DSTLFloatBasic) {
  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({2, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.5f, 1.0f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  Shape output_shape({1, 2, 2, 2});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4DSTL<float>(input, kernel, bias, output, 1, 0, 1, 1);

  std::vector<float> result = *output.as<float>();

  float expected_value = 27.0f;
  ASSERT_NEAR(result[0], expected_value + 0.5f, 1e-5f);
  ASSERT_NEAR(result[4], expected_value + 1.0f, 1e-5f);
}

TEST(ConvolutionalLayerTest, Conv4DSTLFloatWithPaddingAndStride) {
  std::vector<float> image = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                              7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                              13.0f, 14.0f, 15.0f, 16.0f};
  Shape input_shape({1, 1, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 0.0f, 0.0f, 1.0f};
  Shape kernel_shape({1, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (4 + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1;
  size_t out_width = (4 + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1;
  Shape output_shape({1, 1, out_height, out_width});
  std::vector<float> output_vec(
      output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3],
      0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4DSTL<float>(input, kernel, Tensor(), output, 2, 1, 1, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 9);
}

TEST(ConvolutionalLayerTest, Conv4DSTLFloatCompareWithConv4D) {
  std::vector<float> image(27, 1.0f);
  Shape input_shape({1, 3, 3, 3});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(27, 1.0f);
  Shape kernel_shape({1, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape1({1, 1, 1, 1});
  std::vector<float> output_vec1(1, 0.0f);
  Tensor output1 = make_tensor(output_vec1, output_shape1);
  Conv4D<float>(input, kernel, Tensor(), output1, 1, 0, 1, 1);

  Shape output_shape2({1, 1, 1, 1});
  std::vector<float> output_vec2(1, 0.0f);
  Tensor output2 = make_tensor(output_vec2, output_shape2);
  Conv4DSTL<float>(input, kernel, Tensor(), output2, 1, 0, 1, 1);

  float result1 = (*output1.as<float>())[0];
  float result2 = (*output2.as<float>())[0];

  ASSERT_NEAR(result1, result2, 1e-5f);
  ASSERT_NEAR(result1, 27.0f, 1e-5f);
}

TEST(ConvolutionalLayerTest, DepthwiseViaConvolutionalLayer) {
  std::vector<float> image(32, 1.0f);
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(18, 1.0f);
  Shape kernel_shape({2, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape({1, 2, 2, 2});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, Tensor(), kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<float> result = *out[0].as<float>();

  float expected_value = 9.0f;
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], expected_value, 1e-5f);
  }
}

TEST(ConvolutionalLayerTest, Conv4DSTLViaConvolutionalLayer) {
  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({2, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape({1, 2, 2, 2});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, Tensor(), kSTL);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<float> result = *out[0].as<float>();

  float expected_value = 27.0f;
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], expected_value, 1e-5f);
  }
}

TEST(ConvolutionalLayerTest, Conv4DLegacyFloatBasic) {
  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({3, 3, 3, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.5f, 1.0f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4D_Legacy<float>(input, kernel, bias, output, 1, 0, 1);

  std::vector<float> result = *output.as<float>();

  float expected_value_ch1 = 27.0f + 0.5f;
  float expected_value_ch2 = 27.0f + 1.0f;

  ASSERT_EQ(result.size(), 8);
  ASSERT_NEAR(result[0], expected_value_ch1, 1e-5f);
  ASSERT_NEAR(result[1], expected_value_ch1, 1e-5f);
  ASSERT_NEAR(result[4], expected_value_ch2, 1e-5f);
  ASSERT_NEAR(result[5], expected_value_ch2, 1e-5f);
}

TEST(ConvolutionalLayerTest, Conv4DLegacyFloatMultiOutput) {
  std::vector<float> image(32, 1.0f);
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(72, 0.5f);
  Shape kernel_shape({3, 3, 2, 4});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.1f, 0.2f, 0.3f, 0.4f};
  Tensor bias = make_tensor(biasvec, Shape({4}));

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 4, out_height, out_width});
  std::vector<float> output_vec(16, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4D_Legacy<float>(input, kernel, bias, output, 1, 0, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 16);
  ASSERT_NEAR(result[0], 9.0f + 0.1f, 1e-5f);
  ASSERT_NEAR(result[4], 9.0f + 0.2f, 1e-5f);
  ASSERT_NEAR(result[8], 9.0f + 0.3f, 1e-5f);
  ASSERT_NEAR(result[12], 9.0f + 0.4f, 1e-5f);
}

TEST(ConvolutionalLayerTest, Conv4DLegacyViaConvolutionalLayer) {
  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({3, 3, 3, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, Tensor(), kDefault, 1, true);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<float> result = *out[0].as<float>();

  ASSERT_EQ(result.size(), 8);
  float expected_value = 27.0f;
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], expected_value, 1e-5f);
  }
}

TEST(ConvolutionalLayerTest, Conv4DLegacyFloatEdgeCase) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {0.5f};
  Shape kernel_shape({1, 1, 1, 1});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {1.0f};
  Tensor bias = make_tensor(biasvec, Shape({1}));

  size_t out_height = (2 + 2 * 0 - 1 * (1 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (1 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 1, out_height, out_width});
  std::vector<float> output_vec(4, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  Conv4D_Legacy<float>(input, kernel, bias, output, 1, 0, 1);

  std::vector<float> result = *output.as<float>();

  ASSERT_EQ(result.size(), 4);
  ASSERT_NEAR(result[0], 1.0f * 0.5f + 1.0f, 1e-5f);
  ASSERT_NEAR(result[1], 2.0f * 0.5f + 1.0f, 1e-5f);
  ASSERT_NEAR(result[2], 3.0f * 0.5f + 1.0f, 1e-5f);
  ASSERT_NEAR(result[3], 4.0f * 0.5f + 1.0f, 1e-5f);
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DIntPathCoverage) {
  std::vector<int> image = {1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  Shape input_shape({1, 2, 2, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 2, 2, 2, 2};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> biasvec = {10, 20};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<int> output_vec(6, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, bias, kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<int> result = *out[0].as<int>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DFloatPathCoverage) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 1.0f, 1.0f, 1.0f,
                                  0.5f, 0.5f, 0.5f, 0.5f};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.1f, 0.2f};
  Tensor bias = make_tensor(biasvec, Shape({2}));

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(2, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, bias, kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DNoBiasIntPathCoverage) {
  std::vector<int> image = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 2, 2, 2, 2};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<int> output_vec(2, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, Tensor(), kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<int> result = *out[0].as<int>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvolutionalLayerTest, DepthwiseConv4DNoBiasFloatPathCoverage) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 1.0f, 1.0f, 1.0f,
                                  0.5f, 0.5f, 0.5f, 0.5f};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(2, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 0, 1, kernel, Tensor(), kDefault, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvolutionalLayerTest, ConvImplInt2DKernel) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(27, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 3, 3}));

  ConvolutionalLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> result = *out[0].as<int>();
  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 5);
  }
}
TEST(ConvolutionalLayerTest, ConvImplInt2DKernelBasic) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(27, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 3, 3}));

  ConvolutionalLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 5);
  }
}

TEST(ConvolutionalLayerTest, ConvImplInt2DKernelWithStride) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(12, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 2, 2}));

  ConvolutionalLayer layer(2, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 12);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 5);
  }
}

TEST(ConvolutionalLayerTest, ConvImplInt2DKernelWithBias) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> biasvec = {1, 1, 1};
  Tensor bias = make_tensor(biasvec, Shape({3}));
  std::vector<int> output_vec(27, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 3, 3}));

  ConvolutionalLayer layer(1, 0, 1, kernel, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 6);
  }
}

TEST(ConvolutionalLayerTest, ConvImplInt2DKernelSmallInput) {
  std::vector<int> image(27, 2);
  Shape input_shape({1, 3, 3, 3});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  std::vector<int> output_vec(3, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 1, 1}));

  ConvolutionalLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 3);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 18);
  }
}

TEST(ConvolutionalLayerTest, ConvImplInt2DKernelComplexPattern) {
  std::vector<int> image = {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,

                            2, 3, 2, 3, 4, 5, 4, 5, 2, 3, 2, 3, 4, 5, 4, 5,

                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(12, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 2, 2}));

  ConvolutionalLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 12);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_GT(result[i], 0);
  }
}

TEST(ConvolutionalLayerTest, Float2DKernelPathCoverage) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 0.0f, 1.0f, 0.0f};
  Shape kernel_shape({2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> output_vec(1, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({1, 1, 1, 1}));

  ConvolutionalLayer layer(1, 0, 0, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::exception);
}

TEST(ConvolutionalLayerTest, Float4DKernelWorking) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 0.0f, 1.0f, 0.0f};
  Shape kernel_shape({1, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> output_vec(1, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({1, 1, 1, 1}));

  ConvolutionalLayer layer(1, 0, 0, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  ASSERT_EQ(result.size(), 4);
}