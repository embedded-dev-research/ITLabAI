#include <gtest/gtest.h>

#include "layers/EWLayer.hpp"
#include "layers_fused/ConvRelu.hpp"

using namespace it_lab_ai;

TEST(ConvReluLayerTest, CopyFromConv) {
  int step = 2;
  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  std::shared_ptr<ConvolutionalLayer> layer =
      std::make_shared<ConvolutionalLayer>(step, 0, 1, kernel);
  std::vector<float> vec = {1, 2, 3, 4};
  Tensor input1 = make_tensor<float>(vec, {4});
  Tensor input2 = make_tensor<float>(vec, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};
  EXPECT_NO_THROW(ConvReluLayer layer2(layer));
}

TEST(ConvReluLayerTest, IncompatibleInput) {
  int step = 2;
  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<float> vec = {1, 2, 3, 4};
  Tensor input1 = make_tensor<float>(vec, {4});
  Tensor input2 = make_tensor<float>(vec, {2, 2});
  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> output{input1};
  EXPECT_THROW(layer.run(in, output), std::runtime_error);
}

TEST(ConvReluLayerTest, FStep2) {
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
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, FStep1) {
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
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, IntStep2) {
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
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, IntStep1) {
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
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, FloatWithBias) {
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
  ConvReluLayer layer(1, 0, 1, kernel, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, InvalidInputShapeDims) {
  std::vector<float> image(15, 1.0f);
  Shape invalid_shape({1, 3, 5});
  Tensor input = make_tensor(image, invalid_shape);

  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape({1, 3, 3, 3});
  std::vector<float> output_vec(27, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel);

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::out_of_range);
}
TEST(ConvReluLayerTest, Conv4DKern) {
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
  ConvReluLayer layer(step, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}
TEST(ConvReluLayerTest, Conv4DKern_int) {
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
  ConvReluLayer layer(step, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvReluLayerTest, Conv4DKern_int_36) {
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
  ConvReluLayer layer(step, pads, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}

TEST(ConvReluLayerTest, DepthwiseIntegration) {
  std::vector<float> image(32, 1.0f);
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(18, 1.0f);
  Shape kernel_shape({2, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  size_t out_height = (4 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(32, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 1, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  ASSERT_EQ(result.size(), 32);
}

TEST(ConvReluLayerTest, DepthwiseViaConvolutionalLayer) {
  std::vector<float> image(32, -1.0f);
  Shape input_shape({1, 2, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(18, 1.0f);
  Shape kernel_shape({2, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  Shape output_shape({1, 2, 2, 2});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<float> result = *out[0].as<float>();

  float expected_value = 0.0f;
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], expected_value, 1e-5f);
  }
}

TEST(ConvReluLayerTest, Conv4DLegacyViaConvolutionalLayer) {
  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({3, 3, 3, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  size_t out_height = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (4 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel, bias, 1, true);
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

TEST(ConvReluLayerTest, DepthwiseConv4DIntPathCoverage) {
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

  ConvReluLayer layer(1, 0, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<int> result = *out[0].as<int>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvReluLayerTest, DepthwiseConv4DFloatPathCoverage) {
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

  ConvReluLayer layer(1, 0, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvReluLayerTest, DepthwiseConv4DNoBiasIntPathCoverage) {
  std::vector<int> image = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 2, 2, 2, 2};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<int> output_vec(2, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<int> result = *out[0].as<int>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvReluLayerTest, DepthwiseConv4DNoBiasFloatPathCoverage) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Shape input_shape({1, 2, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 1.0f, 1.0f, 1.0f,
                                  0.5f, 0.5f, 0.5f, 0.5f};
  Shape kernel_shape({2, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  size_t out_height = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  size_t out_width = (2 + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1;
  Shape output_shape({1, 2, out_height, out_width});
  std::vector<float> output_vec(2, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel, bias, 2);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  EXPECT_FALSE(result.empty());
}

TEST(ConvReluLayerTest, ConvImplInt2DKernel) {
  std::vector<int> image(75, -1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(27, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 3, 3}));

  ConvReluLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> result = *out[0].as<int>();
  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 0);
  }
}
TEST(ConvReluLayerTest, ConvImplInt2DKernelBasic) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(27, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 3, 3}));

  ConvReluLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 5);
  }
}

TEST(ConvReluLayerTest, ConvImplInt2DKernelWithStride) {
  std::vector<int> image(75, 1);
  Shape input_shape({1, 3, 5, 5});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> output_vec(12, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 2, 2}));

  ConvReluLayer layer(2, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 12);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 5);
  }
}

TEST(ConvReluLayerTest, ConvImplInt2DKernelWithBias) {
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

  ConvReluLayer layer(1, 0, 1, kernel, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 27);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 6);
  }
}

TEST(ConvReluLayerTest, ConvImplInt2DKernelSmallInput) {
  std::vector<int> image(27, 2);
  Shape input_shape({1, 3, 3, 3});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  std::vector<int> output_vec(3, 0);
  Tensor output = make_tensor(output_vec, Shape({1, 3, 1, 1}));

  ConvReluLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 3);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 18);
  }
}

TEST(ConvReluLayerTest, ConvImplInt2DKernelComplexPattern) {
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

  ConvReluLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  layer.run(in, out);

  std::vector<int> result = *out[0].as<int>();

  ASSERT_EQ(result.size(), 12);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_GT(result[i], 0);
  }
}

TEST(ConvReluLayerTest, Float2DKernelPathCoverage) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 0.0f, 1.0f, 0.0f};
  Shape kernel_shape({2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> output_vec(1, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({1, 1, 1, 1}));

  ConvReluLayer layer(1, 0, 0, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::exception);
}

TEST(ConvReluLayerTest, Float4DKernelWorking) {
  std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape input_shape({1, 1, 2, 2});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec = {1.0f, 0.0f, 1.0f, 0.0f};
  Shape kernel_shape({1, 1, 2, 2});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> output_vec(1, 0.0f);
  Tensor output = make_tensor(output_vec, Shape({1, 1, 1, 1}));

  ConvReluLayer layer(1, 0, 0, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  std::vector<float> result = *out[0].as<float>();
  ASSERT_EQ(result.size(), 4);
}

TEST(ConvReluLayerTest, Conv4DWithParallelDefaultFallback) {
  RuntimeOptions options;
  options.backend = Backend::kNaive;

  std::vector<float> image(48, 1.0f);
  Shape input_shape({1, 3, 4, 4});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(54, 1.0f);
  Shape kernel_shape({2, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  Shape output_shape({1, 2, 2, 2});
  std::vector<float> output_vec(8, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvReluLayer layer(1, 0, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out, options);

  std::vector<float> result = *out[0].as<float>();

  float expected_value = 27.0f;
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i], expected_value, 1e-5f);
  }
}
