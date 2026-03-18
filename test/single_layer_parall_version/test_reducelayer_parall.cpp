#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/ReduceLayer.hpp"

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

static Tensor RunReduce(ReduceLayer& layer, const Tensor& input,
                        ParBackend backend) {
  RuntimeOptions options;
  options.par_backend = backend;
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out, options);
  return out[0];
}

static void RunBackendsAndCompare(ReduceLayer& layer, const Tensor& input,
                                  const std::string& label,
                                  float tolerance = 1e-5f) {
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  Tensor baseline = RunReduce(layer, input, ParBackend::kSeq);

  for (auto backend : backends) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor result = RunReduce(layer, input, backend);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    PRINT_TIMING("Reduce " << label << " Backend " << static_cast<int>(backend)
                           << " time: " << duration_ms.count() << " ms");

    ExpectTensorsNear(baseline, result, tolerance);
  }
}

TEST(reducelayer_parall, sum_small_2d_axis0) {
  Shape input_shape({32, 32});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 7);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {0});
  RunBackendsAndCompare(layer, input, "sum_small_2d_axis0");
}

TEST(reducelayer_parall, mean_small_2d_axis1) {
  Shape input_shape({32, 64});
  std::vector<float> input_data(input_shape.count(), 2.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMean, 0, {1});
  RunBackendsAndCompare(layer, input, "mean_small_2d_axis1");
}

TEST(reducelayer_parall, max_small_2d_axis0) {
  Shape input_shape({64, 32});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 50);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMax, 0, {0});
  RunBackendsAndCompare(layer, input, "max_small_2d_axis0");
}

TEST(reducelayer_parall, min_small_3d_axis2) {
  Shape input_shape({8, 16, 16});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 20);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMin, 0, {2});
  RunBackendsAndCompare(layer, input, "min_small_3d_axis2");
}

TEST(reducelayer_parall, int_sum_small_2d_axis0) {
  Shape input_shape({64, 64});
  std::vector<int> input_data(input_shape.count(), 1);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {0});
  RunBackendsAndCompare(layer, input, "int_sum_small_2d_axis0", 0.0f);
}

TEST(reducelayer_parall, sum_big_2d_axis0) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {0});
  RunBackendsAndCompare(layer, input, "sum_big_2d_axis0");
}

TEST(reducelayer_parall, sum_big_2d_axis1) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {1});
  RunBackendsAndCompare(layer, input, "sum_big_2d_axis1");
}

TEST(reducelayer_parall, mean_big_2d_axis0) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count(), 2.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMean, 0, {0});
  RunBackendsAndCompare(layer, input, "mean_big_2d_axis0");
}

TEST(reducelayer_parall, max_big_2d_axis1) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMax, 0, {1});
  RunBackendsAndCompare(layer, input, "max_big_2d_axis1");
}

TEST(reducelayer_parall, min_big_2d_axis0) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count());
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i % 100);
  }
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMin, 0, {0});
  RunBackendsAndCompare(layer, input, "min_big_2d_axis0");
}

TEST(reducelayer_parall, sum_big_3d_axis1) {
  Shape input_shape({64, 256, 256});
  std::vector<float> input_data(input_shape.count(), 1.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {1});
  RunBackendsAndCompare(layer, input, "sum_big_3d_axis1");
}

TEST(reducelayer_parall, sum_big_3d_axis2) {
  Shape input_shape({64, 256, 256});
  std::vector<float> input_data(input_shape.count(), 1.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {2});
  RunBackendsAndCompare(layer, input, "sum_big_3d_axis2");
}

TEST(reducelayer_parall, sum_big_all_axes) {
  Shape input_shape({512, 512});
  std::vector<float> input_data(input_shape.count(), 1.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {0, 1});
  RunBackendsAndCompare(layer, input, "sum_big_all_axes");
}

TEST(reducelayer_parall, mean_big_4d_batch) {
  Shape input_shape({16, 64, 64, 64});
  std::vector<float> input_data(input_shape.count(), 1.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kMean, 0, {0, 2});
  RunBackendsAndCompare(layer, input, "mean_big_4d_batch");
}

TEST(reducelayer_parall, large_tensor_timing) {
  Shape input_shape({128, 512, 512});
  std::vector<float> input_data(input_shape.count(), 1.0f);
  Tensor input = make_tensor(input_data, input_shape);

  ReduceLayer layer(ReduceLayer::Operation::kSum, 0, {0});
  RunBackendsAndCompare(layer, input, "large_tensor_timing");
}
