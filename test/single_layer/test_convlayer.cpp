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
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 2;
  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<float> expected_output(12, 5);
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
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
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 1;
  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<float> expected_output(27, 5);
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
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

  std::vector<float> kernelvec = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  Shape kernel_shape({3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<float> biasvec = {0.5f, 0.5f, 0.5f};
  Tensor bias = make_tensor(biasvec, Shape({3}));

  Shape output_shape({1, 3, 3, 3});
  std::vector<float> output_vec(27, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  std::vector<float> expected_output(27, 5.5f);

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
  Shape sh({2, 2});
  std::vector<float> vec = {1, 2, 3, 4};
  Shape sh1({1, 3, 5, 5});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 1;
  std::vector<float> kernelvec;
  kernelvec.reserve(54);
  for (int i = 0; i < 54; ++i) {
    kernelvec.push_back(1);
  }
  std::vector<float> expected_output(50, 12);
  Shape sh2({3, 3, 3, 2});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> tmp = *out[0].as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}
TEST(ConvolutionalLayerTest, Conv4DKern_int) {
  std::vector<int> image;
  image.reserve(75);
  for (int i = 0; i < 784; ++i) {
    image.push_back(1);
  }
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 1, 28, 28});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 1;
  std::vector<int> kernelvec;
  kernelvec.reserve(54);
  for (int i = 0; i < 400; ++i) {
    kernelvec.push_back(1);
  }
  std::vector<int> expected_output(400 * 16, 25);
  Shape sh2({5, 5, 1, 16});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, 0, 2, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp, expected_output);
}
TEST(ConvolutionalLayerTest, Conv4DKern_int_36) {
  std::vector<int> image;
  image.reserve(75);
  for (int i = 0; i < 16 * 784; ++i) {
    image.push_back(1);
  }
  Shape sh({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh1({1, 16, 28, 28});
  Tensor input = make_tensor(image, sh1);
  Tensor output = make_tensor(vec, sh);
  int step = 1;
  std::vector<int> kernelvec;
  kernelvec.reserve(54);
  for (int i = 0; i < 400 * 36; ++i) {
    kernelvec.push_back(1);
  }
  std::vector<int> expected_output(784 * 36, 0);
  Shape sh2({5, 5, 16, 36});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer layer(step, (kernel.get_shape()[0] - 1) / 2, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> tmp = *out[0].as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
}
