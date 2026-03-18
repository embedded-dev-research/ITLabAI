#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"
#include "layers_oneDNN/PoolingLayer.hpp"

using namespace it_lab_ai;

TEST(poolinglayer_onednn, max_pooling_basic_float) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({1, 1, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {4.0F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(poolinglayer_onednn, max_pooling_basic_int) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  std::vector<int> input_data = {1, 2, 3, 4};
  Tensor input = make_tensor(input_data, Shape({1, 1, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<int>();
  std::vector<int> expected = {4};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(poolinglayer_onednn, average_pooling_basic_float) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                           "average");

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({1, 1, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  std::vector<float> expected = {2.5F};

  ASSERT_EQ(output_data.size(), expected.size());
  for (size_t i = 0; i < output_data.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5);
  }
}

TEST(poolinglayer_onednn, max_pooling_multichannel) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  std::vector<float> input_data(2 * 4 * 4);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input = make_tensor(input_data, Shape({1, 2, 4, 4}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({1, 2, 2, 2}));

  EXPECT_NEAR(output_data[0], 5.0F, 1e-5);
  EXPECT_NEAR(output_data[1], 7.0F, 1e-5);
  EXPECT_NEAR(output_data[2], 13.0F, 1e-5);
  EXPECT_NEAR(output_data[3], 15.0F, 1e-5);
}

TEST(poolinglayer_onednn, max_pooling_with_padding) {
  PoolingLayerOneDnn layer({3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, false, "max");

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Tensor input = make_tensor(input_data, Shape({1, 1, 2, 2}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape, Shape({1, 1, 2, 2}));
}

TEST(poolinglayer_onednn, average_pooling_global) {
  PoolingLayerOneDnn layer({0, 0}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                           "average");

  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                   6.0F, 7.0F, 8.0F, 9.0F};
  Tensor input = make_tensor(input_data, Shape({1, 1, 3, 3}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({1, 1, 1, 1}));

  float expected =
      (1.0F + 2.0F + 3.0F + 4.0F + 5.0F + 6.0F + 7.0F + 8.0F + 9.0F) / 9.0F;
  EXPECT_NEAR(output_data[0], expected, 1e-5);
}

TEST(poolinglayer_onednn, stride_greater_than_kernel) {
  PoolingLayerOneDnn layer({2, 2}, {3, 3}, {0, 0, 0, 0}, {1, 1}, false, "max");

  std::vector<float> input_data(25);
  for (size_t i = 0; i < 25; i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input = make_tensor(input_data, Shape({1, 1, 5, 5}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();

  EXPECT_EQ(output_shape, Shape({1, 1, 2, 2}));
}

TEST(poolinglayer_onednn, dilation) {
  PoolingLayerOneDnn layer({2, 2}, {1, 1}, {0, 0, 0, 0}, {2, 2}, false, "max");

  std::vector<float> input_data(25);
  for (size_t i = 0; i < 25; i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input = make_tensor(input_data, Shape({1, 1, 5, 5}));
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  auto output_data = *out[0].as<float>();
  Shape output_shape = out[0].get_shape();
  EXPECT_EQ(output_shape, Shape({1, 1, 3, 3}));
}

TEST(poolinglayer_onednn, compare_with_naive_max_pooling) {
  PoolingLayerOneDnn onednn_layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                                  "max");
  PoolingLayer naive_layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  std::vector<float> input_data(16);
  for (size_t i = 0; i < 16; i++) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input_tensor = make_tensor(input_data, Shape({1, 1, 4, 4}));

  Tensor onednn_output;
  std::vector<Tensor> onednn_in{input_tensor};
  std::vector<Tensor> onednn_out{onednn_output};
  onednn_layer.run(onednn_in, onednn_out);
  auto onednn_result = *onednn_out[0].as<float>();

  Tensor naive_output;
  std::vector<Tensor> naive_in{input_tensor};
  std::vector<Tensor> naive_out{naive_output};
  naive_layer.run(naive_in, naive_out);
  auto naive_result = *naive_out[0].as<float>();

  ASSERT_EQ(onednn_result.size(), naive_result.size());
  for (size_t i = 0; i < onednn_result.size(); i++) {
    EXPECT_NEAR(onednn_result[i], naive_result[i], 1e-5);
  }
}

TEST(poolinglayer_onednn, invalid_input_tensors) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  Tensor input1 = make_tensor<float>({1.0F, 2.0F});
  Tensor input2 = make_tensor<float>({3.0F, 4.0F});
  Tensor output;

  std::vector<Tensor> in{input1, input2};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(poolinglayer_onednn, invalid_input_dimensions) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  Tensor input = make_tensor<float>({1.0F});
  Tensor output;

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  EXPECT_THROW({ layer.run(in, out); }, std::runtime_error);
}

TEST(poolinglayer_onednn, reinitialization_different_types) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  {
    Tensor input =
        make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
  }

  {
    Tensor input = make_tensor<int>({1, 2, 3, 4}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<int>();
    EXPECT_EQ(result.size(), 1);
  }

  {
    Tensor input =
        make_tensor<float>({5.0F, 6.0F, 7.0F, 8.0F}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
  }
}

TEST(poolinglayer_onednn, different_shapes_same_layer) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  {
    Tensor input =
        make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
  }

  {
    std::vector<float> input_data(16);
    for (size_t i = 0; i < 16; i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({1, 1, 4, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 4);
  }

  {
    Tensor input =
        make_tensor<float>({5.0F, 6.0F, 7.0F, 8.0F}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
  }
}

TEST(poolinglayer_onednn, set_parameters_after_creation) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");

  {
    Tensor input =
        make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({1, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};
    layer.run(in, out);
  }

  layer.setStrides(1, 1);
  layer.setPads(1, 1, 1, 1);
  layer.setCeilMode(true);

  {
    Tensor input = make_tensor<float>(
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F},
        Shape({1, 1, 3, 3}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
  }
}

TEST(poolinglayer_onednn, edge_cases) {
  {
    PoolingLayerOneDnn layer({10, 10}, {1, 1}, {0, 0, 0, 0}, {1, 1}, false,
                             "max");

    std::vector<float> input_data(100);
    for (size_t i = 0; i < 100; i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({1, 1, 10, 10}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    EXPECT_EQ(result.size(), 1);
  }

  {
    PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                             "max");

    std::vector<float> input_data(8);
    for (size_t i = 0; i < 8; i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({2, 1, 2, 2}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto result = *out[0].as<float>();
    Shape output_shape = out[0].get_shape();

    EXPECT_EQ(output_shape, Shape({2, 1, 1, 1}));
    EXPECT_EQ(result.size(), 2);
  }
}

TEST(poolinglayer_onednn, different_input_dimensions) {
  {
    PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                             "max");

    std::vector<float> input_data(1 * 3 * 4 * 4);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({1, 3, 4, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({1, 3, 2, 2}));
  }

  {
    PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                             "max");

    std::vector<float> input_data(3 * 4 * 4);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({3, 4, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({3, 2, 2}));
  }

  {
    PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                             "average");

    std::vector<float> input_data(4 * 4);
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = static_cast<float>(i);
    }

    Tensor input = make_tensor(input_data, Shape({4, 4}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_NO_THROW(layer.run(in, out));
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({2, 2}));
  }
}

TEST(poolinglayer_onednn, invalid_dimensions) {
  PoolingLayerOneDnn layer({2, 2}, {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");
  {
    std::vector<float> input_data(2 * 3 * 4 * 5 * 6);
    Tensor input = make_tensor(input_data, Shape({2, 3, 4, 5, 6}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_THROW(layer.run(in, out), std::runtime_error);
  }

  {
    std::vector<float> input_data(1);
    Tensor input = make_tensor(input_data, Shape({1}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    EXPECT_THROW(layer.run(in, out), std::runtime_error);
  }
}

TEST(poolinglayer_onednn, ceil_mode_with_padding_adjustment) {
  {
    PoolingLayerOneDnn layer({3, 3}, {2, 2}, {0, 0, 0, 0}, {1, 1}, true, "max");

    std::vector<float> input_data(1 * 1 * 5 * 5);
    Tensor input = make_tensor(input_data, Shape({1, 1, 5, 5}));
    Tensor output;
    std::vector<Tensor> in{input};
    std::vector<Tensor> out{output};

    layer.run(in, out);
    auto output_shape = out[0].get_shape();
    EXPECT_EQ(output_shape, Shape({1, 1, 2, 2}));
  }
}
