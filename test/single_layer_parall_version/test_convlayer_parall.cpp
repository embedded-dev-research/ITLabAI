#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"

#define ENABLE_TIMING_OUTPUT 1

#if ENABLE_TIMING_OUTPUT
#  define PRINT_TIMING(msg) std::cout << msg << std::endl
#else
#  define PRINT_TIMING(msg) ((void)0)
#endif

using namespace it_lab_ai;

TEST(convlayer_parall, parallel_conv_basic) {
  size_t batch_size = 32;
  std::vector<float> image(batch_size * 3 * 224 * 224, 1.0f);
  Shape input_shape({batch_size, 3, 224, 224});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(64 * 3 * 3 * 3, 1.0f);
  Shape kernel_shape({64, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (224 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (224 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({batch_size, 64, out_height, out_width});
  std::vector<float> output_vec(batch_size * 64 * out_height * out_width, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("Backend " << static_cast<int>(backend)
                            << " time: " << duration.count()
                            << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    EXPECT_EQ(out[0].get_shape()[1], 64);
  }
}

TEST(convlayer_parall, parallel_conv_stride2) {
  size_t batch_size = 64;
  std::vector<float> image(batch_size * 16 * 112 * 112, 1.0f);
  Shape input_shape({batch_size, 16, 112, 112});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(32 * 16 * 3 * 3, 1.0f);
  Shape kernel_shape({32, 16, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (112 + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1;
  size_t out_width = (112 + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1;
  Shape output_shape({batch_size, 32, out_height, out_width});
  std::vector<float> output_vec(batch_size * 32 * out_height * out_width, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(2, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("Backend " << static_cast<int>(backend)
                            << " time: " << duration.count()
                            << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    EXPECT_EQ(out[0].get_shape()[2], out_height);
    EXPECT_EQ(out[0].get_shape()[3], out_width);
  }
}

TEST(convlayer_parall, parallel_depthwise_conv) {
  size_t batch_size = 128;
  std::vector<float> image(batch_size * 32 * 56 * 56, 1.0f);
  Shape input_shape({batch_size, 32, 56, 56});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(32 * 1 * 3 * 3, 1.0f);
  Shape kernel_shape({32, 1, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);
  Tensor bias;

  size_t out_height = (56 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (56 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({batch_size, 32, out_height, out_width});
  std::vector<float> output_vec(batch_size * 32 * out_height * out_width, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 1, 1, kernel, bias, 32);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("Depthwise Backend " << static_cast<int>(backend)
                                      << " time: " << duration.count()
                                      << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    EXPECT_EQ(out[0].get_shape()[1], 32);
  }
}

TEST(convlayer_parall, parallel_conv_with_bias) {
  size_t batch_size = 16;
  std::vector<int> image(batch_size * 16 * 28 * 28, 1);
  Shape input_shape({batch_size, 16, 28, 28});
  Tensor input = make_tensor(image, input_shape);

  std::vector<int> kernelvec(36 * 16 * 5 * 5, 1);
  Shape kernel_shape({36, 16, 5, 5});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  std::vector<int> biasvec(36, 10);
  Tensor bias = make_tensor(biasvec, Shape({36}));

  size_t pads = (kernel.get_shape()[2] - 1) / 2;
  size_t out_height = (28 + 2 * pads - 1 * (5 - 1) - 1) / 1 + 1;
  size_t out_width = (28 + 2 * pads - 1 * (5 - 1) - 1) / 1 + 1;
  Shape output_shape({batch_size, 36, out_height, out_width});
  std::vector<int> output_vec(batch_size * 36 * out_height * out_width, 0);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, pads, 1, kernel, bias);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("Backend " << static_cast<int>(backend)
                            << " time: " << duration.count()
                            << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    std::vector<int> result = *out[0].as<int>();
    EXPECT_GT(result[0], 0);
  }
}

TEST(convlayer_parall, parallel_conv_large_kernel) {
  size_t batch_size = 8;
  std::vector<float> image(batch_size * 3 * 128 * 128, 1.0f);
  Shape input_shape({batch_size, 3, 128, 128});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(16 * 3 * 7 * 7, 1.0f);
  Shape kernel_shape({16, 3, 7, 7});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t pads = 3;
  size_t out_height = (128 + 2 * pads - 1 * (7 - 1) - 1) / 2 + 1;
  size_t out_width = (128 + 2 * pads - 1 * (7 - 1) - 1) / 2 + 1;
  Shape output_shape({batch_size, 16, out_height, out_width});
  std::vector<float> output_vec(batch_size * 16 * out_height * out_width, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(2, pads, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("7x7 Kernel Backend " << static_cast<int>(backend)
                                       << " time: " << duration.count()
                                       << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    EXPECT_EQ(out[0].get_shape()[1], 16);
  }
}

TEST(convlayer_parall, parallel_conv_single_image) {
  size_t batch_size = 1;
  std::vector<float> image(batch_size * 3 * 224 * 224, 1.0f);
  Shape input_shape({batch_size, 3, 224, 224});
  Tensor input = make_tensor(image, input_shape);

  std::vector<float> kernelvec(64 * 3 * 3 * 3, 1.0f);
  Shape kernel_shape({64, 3, 3, 3});
  Tensor kernel = make_tensor(kernelvec, kernel_shape);

  size_t out_height = (224 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  size_t out_width = (224 + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1;
  Shape output_shape({batch_size, 64, out_height, out_width});
  std::vector<float> output_vec(batch_size * 64 * out_height * out_width, 0.0f);
  Tensor output = make_tensor(output_vec, output_shape);

  ConvolutionalLayer layer(1, 1, 1, kernel);
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING("Single Image Backend " << static_cast<int>(backend)
                                         << " time: " << duration.count()
                                         << " ms (batch=" << batch_size << ")");

    EXPECT_EQ(out[0].get_shape()[0], batch_size);
    EXPECT_EQ(out[0].get_shape()[1], 64);
  }
}
