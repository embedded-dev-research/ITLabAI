#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/BinaryOpLayer.hpp"

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

static Tensor RunBinary(BinaryOpLayer& layer, const Tensor& a, const Tensor& b,
                        ParBackend backend, long long* duration_ms = nullptr) {
  RuntimeOptions options;
  options.par_backend = backend;

  Tensor output;
  std::vector<Tensor> in{a, b};
  std::vector<Tensor> out{output};

  auto start = std::chrono::high_resolution_clock::now();
  layer.run(in, out, options);
  auto end = std::chrono::high_resolution_clock::now();

  if (duration_ms != nullptr) {
    *duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
  }

  return out[0];
}

static void RunAllBackendsAndCompare(BinaryOpLayer& layer, const Tensor& a,
                                     const Tensor& b, const std::string& label,
                                     float tolerance = 1e-5f) {
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  Tensor baseline = RunBinary(layer, a, b, ParBackend::kSeq);

  for (auto backend : backends) {
    long long ms = 0;
    Tensor result = RunBinary(layer, a, b, backend, &ms);

    PRINT_TIMING("BinaryOp " << label << " Backend "
                             << static_cast<int>(backend) << " time: " << ms
                             << " ms");

    ExpectTensorsNear(baseline, result, tolerance);
  }
}

TEST(binaryoplayer_parall, parallel_add_basic_float) {
  Tensor a = make_tensor<float>({1.f, 2.f, 3.f, 4.f}, {2, 2});
  Tensor b = make_tensor<float>({5.f, 6.f, 7.f, 8.f}, {2, 2});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);
  RunAllBackendsAndCompare(layer, a, b, "add_basic_float");
}

TEST(binaryoplayer_parall, parallel_mul_basic_int) {
  Tensor a = make_tensor<int>({1, 2, 3, 4}, {2, 2});
  Tensor b = make_tensor<int>({2, 3, 4, 5}, {2, 2});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  RunAllBackendsAndCompare(layer, a, b, "mul_basic_int", 0.0f);
}

TEST(binaryoplayer_parall, parallel_sub_scalar_float) {
  Shape shape({1024, 1024});
  Tensor a = make_tensor<float>(std::vector<float>(shape.count(), 5.0f), shape);
  Tensor scalar = make_tensor<float>({2.0f});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kSub);
  RunAllBackendsAndCompare(layer, a, scalar, "sub_scalar_float");
}

TEST(binaryoplayer_parall, parallel_div_scalar_float) {
  Shape shape({1024, 1024});
  Tensor a = make_tensor<float>(std::vector<float>(shape.count(), 8.0f), shape);
  Tensor scalar = make_tensor<float>({2.0f});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kDiv);
  RunAllBackendsAndCompare(layer, a, scalar, "div_scalar_float");
}

TEST(binaryoplayer_parall, parallel_broadcast_2d_add) {
  Tensor a = make_tensor<float>(std::vector<float>(1024 * 1, 3.0f), {1024, 1});
  Tensor b = make_tensor<float>(std::vector<float>(1 * 1024, 4.0f), {1, 1024});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);
  RunAllBackendsAndCompare(layer, a, b, "broadcast_2d_add");
}

TEST(binaryoplayer_parall, parallel_broadcast_3d_mul) {
  Tensor a =
      make_tensor<float>(std::vector<float>(64 * 1 * 512, 1.5f), {64, 1, 512});
  Tensor b =
      make_tensor<float>(std::vector<float>(64 * 512 * 1, 2.0f), {64, 512, 1});

  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  RunAllBackendsAndCompare(layer, a, b, "broadcast_3d_mul");
}

TEST(binaryoplayer_parall, parallel_large_add_same_shape) {
  Shape shape({2048, 2048});
  Tensor a = make_tensor<float>(std::vector<float>(shape.count(), 1.0f), shape);
  Tensor b = make_tensor<float>(std::vector<float>(shape.count(), 2.0f), shape);

  BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);
  RunAllBackendsAndCompare(layer, a, b, "large_add_same_shape");
}

TEST(binaryoplayer_parall, parallel_large_mul_same_shape) {
  Shape shape({1024, 1024, 4});
  Tensor a =
      make_tensor<float>(std::vector<float>(shape.count(), 1.25f), shape);
  Tensor b = make_tensor<float>(std::vector<float>(shape.count(), 2.0f), shape);

  BinaryOpLayer layer(BinaryOpLayer::Operation::kMul);
  RunAllBackendsAndCompare(layer, a, b, "large_mul_same_shape");
}

TEST(binaryoplayer_parall, parallel_large_broadcast_4d_add) {
  Shape a_shape({16, 32, 128, 1});
  Shape b_shape({1, 32, 1, 128});

  Tensor a =
      make_tensor<float>(std::vector<float>(a_shape.count(), 1.0f), a_shape);
  Tensor b =
      make_tensor<float>(std::vector<float>(b_shape.count(), 2.0f), b_shape);

  BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);
  RunAllBackendsAndCompare(layer, a, b, "large_broadcast_4d_add");
}

// TEST(binaryoplayer_parall, parallel_huge_timing_add) {
//   Shape shape({128, 512, 512});
//   Tensor a = make_tensor<float>(std::vector<float>(shape.count(), 1.0f),
//   shape); Tensor b =
//   make_tensor<float>(std::vector<float>(shape.count(), 2.0f), shape);

//   BinaryOpLayer layer(BinaryOpLayer::Operation::kAdd);
//   RunAllBackendsAndCompare(layer, a, b, "huge_timing_add");
// }
