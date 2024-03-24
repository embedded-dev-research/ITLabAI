#include <gtest/gtest.h>

#include "layers/ConvLayer.hpp"

TEST(ConvolutionalLayerTest, Step2) {
  std::vector<float> image;
  for (float i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh1({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh({1, 5, 5, 3});
  Tensor input = make_tensor(image, sh);
  Tensor output = make_tensor(vec, sh1);
  int step = 2;
  std::vector<float> kernel = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<float> expected_output(12, 5);
  ConvolutionalLayer<float> conv_layer;
  conv_layer.run(input, output, kernel, step);
  std::vector<float> tmp = *output.as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, Step1) {
  std::vector<float> image;
  for (float i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  Shape sh1({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Shape sh({1, 5, 5, 3});
  Tensor input = make_tensor(image, sh);
  Tensor output = make_tensor(vec, sh1);
  int step = 1;
  std::vector<float> kernel = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<float> expected_output(27, 5);
  ConvolutionalLayer<float> conv_layer;
  conv_layer.run(input, output, kernel, step);
  std::vector<float> tmp = *output.as<float>();
  ASSERT_EQ(tmp.size(), expected_output.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    ASSERT_FLOAT_EQ(tmp[i], expected_output[i]);
  }
}