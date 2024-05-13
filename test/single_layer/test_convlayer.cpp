#include <gtest/gtest.h>

#include "layers/ConvLayer.hpp"

using namespace itlab_2023;

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
  ConvolutionalLayer layer(step, 0, 0, kernel);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
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
  ConvolutionalLayer layer(step, 0, 0, kernel);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
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
  ConvolutionalLayer layer(step, 0, 0, kernel);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
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
  ConvolutionalLayer layer(step, 0, 0, kernel);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_EQ(tmp[i], expected_output[i]);
  }
}