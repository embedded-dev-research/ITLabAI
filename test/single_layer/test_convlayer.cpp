#include <gtest/gtest.h>

#include "layers/ConvLayer.hpp"

TEST(ConvolutionalLayerTest, Step2) {
  std::vector<float> image;
  for (float i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  size_t image_width = 5;
  size_t image_height = 5;
  int step = 2;
  std::vector<float> kernel = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<size_t> outsize;
  outsize.push_back(2);
  outsize.push_back(2);
  Shape kernel_shape(outsize);
  std::vector<float> expected_output (12, 5);
  ConvolutionalLayer<float> conv_layer(kernel, kernel_shape, image_width,
                                       image_height, step);
  std::vector<float> output = conv_layer.run(image);
  ASSERT_EQ(output.size(), expected_output.size());
  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_FLOAT_EQ(output[i], expected_output[i]);
  }
}
TEST(ConvolutionalLayerTest, Step1) {
  std::vector<float> image;
  for (float i = 0; i < 75; ++i) {
    image.push_back(1);
  }
  size_t image_width = 5;
  size_t image_height = 5;
  int step = 1;
  std::vector<float> kernel = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<size_t> outsize;
  outsize.push_back(2);
  outsize.push_back(2);
  Shape kernel_shape(outsize);
  std::vector<float> expected_output(27, 5);
  ConvolutionalLayer<float> conv_layer(kernel, kernel_shape, image_width,
                                       image_height, step);
  std::vector<float> output = conv_layer.run(image);
  ASSERT_EQ(output.size(), expected_output.size());
  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_FLOAT_EQ(output[i], expected_output[i]);
  }
}
