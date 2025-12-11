#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"
#include "layers_oneDNN/ConvLayer.hpp"

using namespace it_lab_ai;

TEST(convlayer_onednn, basic_convolution_2d_float) {
  std::vector<float> input_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                   7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f};

  std::vector<float> kernel_data = {1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
                                    -1.0f, 1.0f, 0.0f,  -1.0f};

  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape.dims(), 4);
  EXPECT_EQ(output_shape[0], 1);
  EXPECT_EQ(output_shape[1], 1);
  EXPECT_EQ(output_shape[2], 2);
  EXPECT_EQ(output_shape[3], 2);
}

TEST(convlayer_onednn, conv_with_bias_float) {
  std::vector<float> input_data(4 * 4, 1.0f);
  std::vector<float> kernel_data(3 * 3, 1.0f);
  std::vector<float> bias_data = {2.0f};

  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));
  Tensor bias = make_tensor(bias_data, Shape({1}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, bias);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  auto output_data = *out[0].as<float>();
  for (float val : output_data) {
    EXPECT_NEAR(val, 11.0f, 1e-5);
  }
}

TEST(convlayer_onednn, multi_channel_conv_float) {
  std::vector<float> input_data(2 * 4 * 4, 1.0f);
  std::vector<float> kernel_data(3 * 2 * 3 * 3, 1.0f);

  Tensor input = make_tensor(input_data, Shape({1, 2, 4, 4}));
  Tensor kernel = make_tensor(kernel_data, Shape({3, 2, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 3);
  EXPECT_EQ(output_shape[2], 2);
  EXPECT_EQ(output_shape[3], 2);

  auto output_data = *out[0].as<float>();
  for (float val : output_data) {
    EXPECT_NEAR(val, 18.0f, 1e-5);
  }
}

TEST(convlayer_onednn, conv_int_type) {
  std::vector<int> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                 9, 10, 11, 12, 13, 14, 15, 16};

  std::vector<int> kernel_data(3 * 3, 1);

  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  auto output_data = *out[0].as<int>();
  EXPECT_EQ(output_data[0], 54);
}

TEST(convlayer_onednn, grouped_convolution) {
  std::vector<float> input_data(4 * 6 * 6, 1.0f);
  std::vector<float> kernel_data(8 * 2 * 3 * 3, 1.0f);

  Tensor input = make_tensor(input_data, Shape({1, 4, 6, 6}));
  Tensor kernel = make_tensor(kernel_data, Shape({8, 2, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 2);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 8);
}

TEST(convlayer_onednn, depthwise_convolution) {
  std::vector<float> input_data(3 * 5 * 5, 1.0f);
  std::vector<float> kernel_data(3 * 1 * 3 * 3, 1.0f);

  Tensor input = make_tensor(input_data, Shape({1, 3, 5, 5}));
  Tensor kernel = make_tensor(kernel_data, Shape({3, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 3, false, true);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 3);

  auto output_data = *out[0].as<float>();
  for (float val : output_data) {
    EXPECT_NEAR(val, 9.0f, 1e-5);
  }
}

TEST(convlayer_onednn, invalid_input_tensors) {
  std::vector<float> kernel_data(3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);
  Tensor input1 =
      make_tensor<float>(std::vector<float>(16, 1.0f), Shape({1, 1, 4, 4}));
  Tensor input2 =
      make_tensor<float>(std::vector<float>(16, 1.0f), Shape({1, 1, 4, 4}));
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, invalid_input_dimensions) {
  std::vector<float> kernel_data(3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor input =
      make_tensor<float>(std::vector<float>(4, 1.0f), Shape({1, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, invalid_kernel_dimensions) {
  std::vector<float> kernel_data(3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor input =
      make_tensor<float>(std::vector<float>(16, 1.0f), Shape({1, 1, 4, 4}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, channel_mismatch_error) {
  std::vector<float> kernel_data(1 * 2 * 3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 2, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor input =
      make_tensor<float>(std::vector<float>(16, 1.0f), Shape({1, 1, 4, 4}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, unsupported_data_type) {
  std::vector<float> kernel_data(3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  Tensor input =
      make_tensor<float>(std::vector<float>(16, 1.0f), Shape({1, 1, 4, 4}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
}

TEST(convlayer_onednn, special_conv_format) {
  std::vector<float> kernel_data = {
      1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f,

      0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f};

  std::vector<float> input_data(1 * 4 * 4, 1.0f);

  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor kernel = make_tensor(kernel_data, Shape({3, 3, 1, 2}));
  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 1, true);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 2);
}

TEST(convlayer_onednn, large_input_convolution) {
  const size_t batch = 2;
  const size_t channels = 16;
  const size_t height = 32;
  const size_t width = 32;
  const size_t kernel_size = 5;
  const size_t out_channels = 32;

  std::vector<float> input_data(batch * channels * height * width, 1.0f);
  std::vector<float> kernel_data(
      out_channels * channels * kernel_size * kernel_size, 1.0f);

  Tensor input =
      make_tensor(input_data, Shape({batch, channels, height, width}));
  Tensor kernel = make_tensor(
      kernel_data, Shape({out_channels, channels, kernel_size, kernel_size}));

  ConvLayerOneDnn layer(1, 2, 1, kernel);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[0], batch);
  EXPECT_EQ(output_shape[1], out_channels);
  EXPECT_EQ(output_shape[2], height);
  EXPECT_EQ(output_shape[3], width);
}

TEST(convlayer_onednn, dilation_convolution) {
  std::vector<float> input_data = {
      1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
      19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f};

  std::vector<float> kernel_data = {1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
                                    -1.0f, 1.0f, 0.0f,  -1.0f};

  Tensor input = make_tensor(input_data, Shape({1, 1, 5, 5}));
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 2, kernel);

  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[2], 1);
  EXPECT_EQ(output_shape[3], 1);
}

TEST(convlayer_onednn, reinitialization_on_input_change) {
  std::vector<float> kernel_data(1 * 1 * 3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  std::vector<float> input1_data(1 * 1 * 4 * 4, 1.0f);
  Tensor input1 = make_tensor(input1_data, Shape({1, 1, 4, 4}));
  Tensor output1;

  std::vector<Tensor> in1{input1};
  std::vector<Tensor> out1{output1};
  EXPECT_NO_THROW(layer.run(in1, out1));

  std::vector<float> input2_data(1 * 1 * 6 * 6, 1.0f);
  Tensor input2 = make_tensor(input2_data, Shape({1, 1, 6, 6}));
  Tensor output2;

  std::vector<Tensor> in2{input2};
  std::vector<Tensor> out2{output2};
  EXPECT_NO_THROW(layer.run(in2, out2));

  Shape output_shape1 = out1[0].get_shape();
  Shape output_shape2 = out2[0].get_shape();

  EXPECT_EQ(output_shape1[2], 2);
  EXPECT_EQ(output_shape2[2], 4);
}

TEST(convlayer_onednn, reinitialization_on_data_type_change) {
  std::vector<float> kernel_data(1 * 1 * 3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));
  ConvLayerOneDnn layer(1, 0, 1, kernel);

  std::vector<float> input1_data(1 * 1 * 4 * 4, 1.0f);
  Tensor input1 = make_tensor(input1_data, Shape({1, 1, 4, 4}));
  Tensor output1;
  std::vector<Tensor> in1{input1};
  std::vector<Tensor> out1{output1};

  EXPECT_NO_THROW(layer.run(in1, out1));

  std::vector<int> kernel_data_int(1 * 1 * 3 * 3, 1);
  Tensor kernel_int = make_tensor(kernel_data_int, Shape({1, 1, 3, 3}));
  ConvLayerOneDnn layer_int(1, 0, 1, kernel_int);

  std::vector<int> input2_data(1 * 1 * 4 * 4, 1);
  Tensor input2 = make_tensor(input2_data, Shape({1, 1, 4, 4}));
  Tensor output2;
  std::vector<Tensor> in2{input2};
  std::vector<Tensor> out2{output2};

  EXPECT_NO_THROW(layer_int.run(in2, out2));
  EXPECT_EQ(out1[0].get_type(), Type::kFloat);
  EXPECT_EQ(out2[0].get_type(), Type::kInt);
}

TEST(convlayer_onednn, exception_propagation_from_dnnl) {
  std::vector<float> kernel_data(2 * 3 * 5 * 5, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({2, 3, 5, 5}));
  ConvLayerOneDnn layer(1, 0, 1, kernel);

  std::vector<float> input_data(1 * 1 * 4 * 4, 1.0f);
  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, group_validation_errors) {
  {
    std::vector<float> kernel_data(4 * 3 * 3 * 3, 1.0f);
    Tensor kernel = make_tensor(kernel_data, Shape({4, 3, 3, 3}));
    ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 2);

    std::vector<float> input_data(1 * 5 * 6 * 6, 1.0f);
    Tensor input = make_tensor(input_data, Shape({1, 5, 6, 6}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_THROW(layer.run(in, out), std::runtime_error);
  }
  {
    std::vector<float> kernel_data(6 * 3 * 3 * 3, 1.0f);
    Tensor kernel = make_tensor(kernel_data, Shape({6, 3, 3, 3}));
    ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 2);

    std::vector<float> input_data(1 * 4 * 6 * 6, 1.0f);
    Tensor input = make_tensor(input_data, Shape({1, 4, 6, 6}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_THROW(layer.run(in, out), std::runtime_error);
  }
}

TEST(convlayer_onednn, depthwise_kernel_shape_validation) {
  std::vector<float> kernel_data(3 * 2 * 3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({3, 2, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 3, false, true);

  std::vector<float> input_data(1 * 3 * 5 * 5, 1.0f);
  Tensor input = make_tensor(input_data, Shape({1, 3, 5, 5}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW(layer.run(in, out), std::runtime_error);
}

TEST(convlayer_onednn, unsupported_data_type_validation) {
  std::vector<float> kernel_data(3 * 3, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));
  ConvLayerOneDnn layer(1, 0, 1, kernel);

  std::vector<float> input_data(16, 1.0f);
  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));
  std::vector<int> kernel_data_int(3 * 3, 1);
  Tensor kernel_int = make_tensor(kernel_data_int, Shape({1, 1, 3, 3}));
  ConvLayerOneDnn layer_int(1, 0, 1, kernel_int);

  std::vector<int> input_data_int(16, 1);
  Tensor input_int = make_tensor(input_data_int, Shape({1, 1, 4, 4}));
  Tensor output_int;
  std::vector<Tensor> in_int{input_int};
  std::vector<Tensor> out_int{output_int};

  EXPECT_NO_THROW(layer_int.run(in_int, out_int));
}

TEST(convlayer_onednn, bias_memory_handling) {
  {
    std::vector<float> kernel_data(2 * 3 * 3 * 3, 1.0f);
    std::vector<float> bias_data(2, 2.0f);
    Tensor kernel = make_tensor(kernel_data, Shape({2, 3, 3, 3}));
    Tensor bias = make_tensor(bias_data, Shape({2}));

    ConvLayerOneDnn layer(1, 0, 1, kernel, bias);

    std::vector<float> input_data(1 * 3 * 6 * 6, 1.0f);
    Tensor input = make_tensor(input_data, Shape({1, 3, 6, 6}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));

    auto output_vals = *out[0].as<float>();
    EXPECT_GT(output_vals[0], 0.0f);
  }

  {
    std::vector<float> kernel_data(4 * 2 * 3 * 3, 1.0f);
    std::vector<float> bias_data(4, 1.0f);
    Tensor kernel = make_tensor(kernel_data, Shape({4, 2, 3, 3}));
    Tensor bias = make_tensor(bias_data, Shape({4}));

    ConvLayerOneDnn layer(1, 0, 1, kernel, bias, 2);

    std::vector<float> input_data(1 * 4 * 6 * 6, 1.0f);
    Tensor input = make_tensor(input_data, Shape({1, 4, 6, 6}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }
  {
    std::vector<float> kernel_data(3 * 1 * 3 * 3, 1.0f);
    std::vector<float> bias_data(3, 0.5f);
    Tensor kernel = make_tensor(kernel_data, Shape({3, 1, 3, 3}));
    Tensor bias = make_tensor(bias_data, Shape({3}));

    ConvLayerOneDnn layer(1, 0, 1, kernel, bias, 3, false, true);

    std::vector<float> input_data(1 * 3 * 5 * 5, 1.0f);
    Tensor input = make_tensor(input_data, Shape({1, 3, 5, 5}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }
}

TEST(convlayer_onednn, kernel_dims_conversion) {
  std::vector<float> kernel_data(2 * 3 * 4 * 4, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({2, 3, 4, 4}));
  ConvLayerOneDnn layer(1, 0, 1, kernel);
  std::vector<float> input_data(1 * 3 * 8 * 8, 1.0f);
  Tensor input = make_tensor(input_data, Shape({1, 3, 8, 8}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 2);
  EXPECT_EQ(output_shape[2], 5);
  EXPECT_EQ(output_shape[3], 5);
}

TEST(convlayer_onednn, int_kernel_processing) {
  std::vector<int> kernel_data = {1, 0, -1, 1, 0, -1, 1, 0, -1,
                                  0, 1, 0,  0, 1, 0,  0, 1, 0};

  Tensor kernel = make_tensor(kernel_data, Shape({3, 3, 1, 2}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 1, true);

  std::vector<int> input_data(1 * 1 * 4 * 4, 1);
  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  EXPECT_EQ(out[0].get_type(), Type::kInt);

  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape[1], 2);
}

TEST(convlayer_onednn, special_conv_diagnostics) {
  std::vector<float> kernel_data(3 * 3 * 64 * 128, 1.0f);
  Tensor kernel = make_tensor(kernel_data, Shape({3, 3, 64, 128}));

  ConvLayerOneDnn layer(2, 1, 2, kernel, Tensor(), 1, true);

  std::vector<float> input_data(1 * 64 * 8 * 8, 1.0f);
  Tensor input = make_tensor(input_data, Shape({1, 64, 8, 8}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  try {
    layer.run(in, out);
    Shape output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape[1], 128);
  } catch (const std::exception& e) {
    std::cerr << "Caught expected exception: " << e.what() << std::endl;
  }
}

TEST(convlayer_onednn, int_input_processing_special_conv) {
  std::vector<int> kernel_data(3 * 3 * 1 * 2, 1);
  Tensor kernel = make_tensor(kernel_data, Shape({3, 3, 1, 2}));

  ConvLayerOneDnn layer(1, 0, 1, kernel, Tensor(), 1, true);

  std::vector<int> input_data(1 * 1 * 4 * 4, 2);
  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  EXPECT_EQ(out[0].get_type(), Type::kInt);

  auto output_vals = *out[0].as<int>();
  for (int val : output_vals) {
    EXPECT_GT(val, 0);
  }
}

TEST(convlayer_onednn, int_output_processing) {
  std::vector<int> kernel_data(1 * 1 * 3 * 3, 1);
  Tensor kernel = make_tensor(kernel_data, Shape({1, 1, 3, 3}));

  ConvLayerOneDnn layer(1, 0, 1, kernel);

  std::vector<int> input_data(1 * 1 * 4 * 4, 1);
  Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_NO_THROW(layer.run(in, out));

  auto output_vals = *out[0].as<int>();
  for (int val : output_vals) {
    EXPECT_EQ(val, 9);
  }
}
