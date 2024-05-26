#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "layers/FCLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "perf/benchmarking.hpp"

using namespace itlab_2023;

void test_func(PoolingLayer& p, const Tensor& input, Tensor& output) {
  p.run(input, output);
}

TEST(time_test, mat_vec_mul_comp) {
  size_t k = 5000;
  std::vector<int> mat(k * k);
  std::vector<int> vec(k);
  for (size_t i = 0; i < k; i++) {
    vec[i] = rand();
  }
  for (size_t i = 0; i < k * k; i++) {
    mat[i] = rand();
  }
  double count1 = elapsed_time_avg<double, std::milli>(500, mat_vec_mul<int>,
                                                       mat, Shape({k, k}), vec);
  std::cerr << "Normal:" << count1 << std::endl;
  double count2 = elapsed_time_avg<double, std::milli>(
      500, mat_vec_mul_tbb<int>, mat, Shape({k, k}), vec);
  std::cerr << "Tbb:" << count2 << std::endl;
  EXPECT_GE(count1, count2);
}

TEST(pooling_test, is_parallel_good) {
  size_t n = 1000;
  size_t c = 3;
  size_t h = 224;
  size_t w = 224;
  Shape test_shape = {n, c, h, w};
  std::vector<int> a1(n * c * h * w);
  for (size_t i = 0; i < n * c * h * w; i++) {
    a1[i] = rand();
  }
  Tensor input = make_tensor(a1, test_shape);
  Tensor output;
  PoolingLayer p1(Shape({2, 2}), "max", kDefault);
  PoolingLayer p2(Shape({2, 2}), "max", kTBB);
  double count1 =
      elapsed_time<double, std::milli>(test_func, p1, input, output);
  std::cerr << "Normal:" << count1 << std::endl;
  double count2 =
      elapsed_time<double, std::milli>(test_func, p2, input, output);
  std::cerr << "Tbb:" << count2 << std::endl;
}
