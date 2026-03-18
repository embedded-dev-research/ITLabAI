#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "layers/PoolingLayer.hpp"

#define ENABLE_TIMING_OUTPUT 1

#if ENABLE_TIMING_OUTPUT
#  define PRINT_TIMING(msg) std::cout << msg << std::endl
#else
#  define PRINT_TIMING(msg) ((void)0)
#endif

using namespace it_lab_ai;

static void ExpectTensorsNear(const Tensor& a, const Tensor& b,
                              float tolerance = 1e-5f) {
  ASSERT_EQ(a.get_shape(), b.get_shape());
  ASSERT_EQ(a.get_type(), b.get_type());

  if (a.get_type() == Type::kFloat) {
    auto data_a = *a.as<float>();
    auto data_b = *b.as<float>();
    ASSERT_EQ(data_a.size(), data_b.size());
    for (size_t i = 0; i < data_a.size(); ++i) {
      EXPECT_NEAR(data_a[i], data_b[i], tolerance) << "Mismatch at index " << i;
    }
  } else if (a.get_type() == Type::kInt) {
    auto data_a = *a.as<int>();
    auto data_b = *b.as<int>();
    ASSERT_EQ(data_a.size(), data_b.size());
    for (size_t i = 0; i < data_a.size(); ++i) {
      EXPECT_EQ(data_a[i], data_b[i]) << "Mismatch at index " << i;
    }
  }
}

static Tensor RunPooling(PoolingLayer& layer, const Tensor& input,
                         ParBackend backend) {
  RuntimeOptions options;
  options.par_backend = backend;
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out, options);
  return out[0];
}

TEST(poolinglayer_parall, max_pooling_single_image_float) {
  Shape input_shape({1, 1, 4, 4});
  std::vector<float> input_data = {1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,
                                   7.0F,  8.0F,  9.0F,  10.0F, 11.0F, 12.0F,
                                   13.0F, 14.0F, 15.0F, 16.0F};
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");
  Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    Tensor result = RunPooling(layer, input, backend);
    ExpectTensorsNear(baseline, result, 1e-5F);
  }
}

TEST(poolinglayer_parall, avg_pooling_single_image_float) {
  Shape input_shape({1, 1, 4, 4});
  std::vector<float> input_data = {1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,
                                   7.0F,  8.0F,  9.0F,  10.0F, 11.0F, 12.0F,
                                   13.0F, 14.0F, 15.0F, 16.0F};
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                     "average");
  Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    Tensor result = RunPooling(layer, input, backend);
    ExpectTensorsNear(baseline, result, 1e-5F);
  }
}

TEST(poolinglayer_parall, max_pooling_single_image_int) {
  Shape input_shape({1, 1, 4, 4});
  std::vector<int> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                 9, 10, 11, 12, 13, 14, 15, 16};
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");
  Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    Tensor result = RunPooling(layer, input, backend);
    ExpectTensorsNear(baseline, result, 0.0F);
  }
}

TEST(poolinglayer_parall, pooling_with_padding_and_stride) {
  Shape input_shape({1, 3, 8, 8});
  std::vector<float> input_data(input_shape.count(), 2.5F);
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({3, 3}), {2, 2}, {1, 1, 1, 1}, {1, 1}, false,
                     "average");
  Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    Tensor result = RunPooling(layer, input, backend);
    ExpectTensorsNear(baseline, result, 1e-5F);
  }
}

TEST(poolinglayer_parall, pooling_with_dilation) {
  Shape input_shape({1, 1, 8, 8});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({2, 2}), {1, 1}, {0, 0, 0, 0}, {2, 2}, false, "max");
  Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    Tensor result = RunPooling(layer, input, backend);
    ExpectTensorsNear(baseline, result, 1e-5F);
  }
}

TEST(poolinglayer_parall, max_pooling_batch_scaling) {
  std::vector<size_t> batch_sizes = {1, 16, 32, 64};
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (size_t batch_size : batch_sizes) {
    Shape input_shape({batch_size, 16, 56, 56});
    std::vector<float> input_data(input_shape.count(), 1.5F);
    Tensor input = make_tensor(input_data, input_shape);

    PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                       "max");

    Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

    for (auto backend : backends) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor result = RunPooling(layer, input, backend);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      PRINT_TIMING("MaxPool Backend " << static_cast<int>(backend)
                                      << " time: " << duration_ms.count()
                                      << " ms (batch=" << batch_size << ")");

      ExpectTensorsNear(baseline, result, 1e-5F);
    }
  }
}

TEST(poolinglayer_parall, avg_pooling_batch_scaling) {
  std::vector<size_t> batch_sizes = {1, 16, 32, 64};
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (size_t batch_size : batch_sizes) {
    Shape input_shape({batch_size, 32, 28, 28});
    std::vector<float> input_data(input_shape.count(), 2.0F);
    Tensor input = make_tensor(input_data, input_shape);

    PoolingLayer layer(Shape({3, 3}), {2, 2}, {1, 1, 1, 1}, {1, 1}, false,
                       "average");

    Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

    for (auto backend : backends) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor result = RunPooling(layer, input, backend);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      PRINT_TIMING("AvgPool Backend " << static_cast<int>(backend)
                                      << " time: " << duration_ms.count()
                                      << " ms (batch=" << batch_size << ")");

      ExpectTensorsNear(baseline, result, 1e-5F);
    }
  }
}

TEST(poolinglayer_parall, multichannel_max_pooling) {
  std::vector<size_t> batch_sizes = {8, 32, 128};
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (size_t batch_size : batch_sizes) {
    Shape input_shape({batch_size, 64, 14, 14});
    std::vector<int> input_data(input_shape.count());
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 100);
    for (auto& val : input_data) {
      val = dist(rng);
    }
    Tensor input = make_tensor(input_data, input_shape);

    PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false,
                       "max");

    Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

    for (auto backend : backends) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor result = RunPooling(layer, input, backend);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      PRINT_TIMING("MultiChannel MaxPool Backend "
                   << static_cast<int>(backend)
                   << " time: " << duration_ms.count()
                   << " ms (batch=" << batch_size << ", channels=64)");

      ExpectTensorsNear(baseline, result, 0.0F);
    }
  }
}

TEST(poolinglayer_parall, large_kernel_pooling) {
  std::vector<size_t> batch_sizes = {4, 16, 32};
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (size_t batch_size : batch_sizes) {
    Shape input_shape({batch_size, 3, 224, 224});
    std::vector<float> input_data(input_shape.count(), 1.0F);
    Tensor input = make_tensor(input_data, input_shape);

    PoolingLayer layer(Shape({7, 7}), {4, 4}, {2, 2, 2, 2}, {1, 1}, false,
                       "average");

    Tensor baseline = RunPooling(layer, input, ParBackend::kSeq);

    for (auto backend : backends) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor result = RunPooling(layer, input, backend);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      PRINT_TIMING("LargeKernel AvgPool Backend "
                   << static_cast<int>(backend)
                   << " time: " << duration_ms.count()
                   << " ms (batch=" << batch_size << ")");

      ExpectTensorsNear(baseline, result, 1e-4F);
    }
  }
}

TEST(poolinglayer_parall, 1d_pooling_correctness) {
  Shape input_shape({8});
  std::vector<float> input_data = {9.0F, 8.0F, 7.0F, 6.0F,
                                   5.0F, 4.0F, 3.0F, 2.0F};
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({3}), {2}, {0, 0, 0, 0}, {1, 1}, false, "average");
  Tensor result = RunPooling(layer, input, ParBackend::kSeq);

  auto output = *result.as<float>();
  std::vector<float> expected = {8.0F, 6.0F, 4.0F};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5F);
  }
}

TEST(poolinglayer_parall, 2d_pooling_with_specified_values) {
  Shape input_shape({1, 1, 4, 4});
  std::vector<float> input_data = {1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,
                                   7.0F,  8.0F,  9.0F,  10.0F, 11.0F, 12.0F,
                                   13.0F, 14.0F, 15.0F, 16.0F};
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({2, 2}), {2, 2}, {0, 0, 0, 0}, {1, 1}, false, "max");
  Tensor result = RunPooling(layer, input, ParBackend::kSeq);

  auto output = *result.as<float>();
  std::vector<float> expected = {6.0F, 8.0F, 14.0F, 16.0F};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5F);
  }
}

TEST(poolinglayer_parall, global_avg_pooling) {
  Shape input_shape({2, 3, 5, 5});
  std::vector<float> input_data(input_shape.count(), 4.0F);
  Tensor input = make_tensor(input_data, input_shape);

  PoolingLayer layer(Shape({0, 0}), {1, 1}, {0, 0, 0, 0}, {1, 1}, false,
                     "average");
  Tensor result = RunPooling(layer, input, ParBackend::kSeq);

  auto output = *result.as<float>();
  ASSERT_EQ(output.size(), 2 * 3);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], 4.0F, 1e-5F);
  }
}
