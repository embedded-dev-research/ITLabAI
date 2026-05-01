#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

void test_func(Layer& p, const Tensor& input, Tensor& output,
               const RuntimeOptions& options) {
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  p.run(in, out, options);
}

TEST(pooling_test, is_pooling_tbb_ok) {
  size_t n = 10;
  size_t c = 3;
  size_t h = 224;
  size_t w = 224;
  Shape test_shape = {n, c, h, w};
  std::vector<int> a1(n * c * h * w);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
  for (size_t i = 0; i < n * c * h * w; i++) {
    a1[i] = dist(rng);
  }
  Tensor input = make_tensor(a1, test_shape);
  Tensor output;
  RuntimeOptions options_seq;
  options_seq.par_backend = ParBackend::kSeq;

  RuntimeOptions options_tbb;
  options_tbb.par_backend = ParBackend::kTbb;

  PoolingLayer p1(Shape({2, 2}), "max");
  PoolingLayer p2(Shape({2, 2}), "max");
  double count1 = elapsed_time<double, std::milli>(test_func, p1, input, output,
                                                   options_seq);
  double count2 = elapsed_time<double, std::milli>(test_func, p2, input, output,
                                                   options_tbb);
  std::cout << count1 << " vs. " << count2 << " (parallel)\n";

#ifdef ITLABAI_HAS_SYCL
  RuntimeOptions options_sycl;
  options_sycl.par_backend = ParBackend::kSycl;
  PoolingLayer p3(Shape({2, 2}), "max");
  double count3 = elapsed_time<double, std::milli>(test_func, p3, input, output,
                                                   options_sycl);
  std::cout << count1 << " vs. " << count3 << " (sycl)\n";
#endif
}

TEST(conv_test, is_conv_stl_ok) {
  size_t n = 10;
  size_t c = 3;
  size_t h = 224;
  size_t w = 224;
  Shape test_shape = {n, c, h, w};
  std::vector<int> a1(n * c * h * w);
  std::vector<int> a2(3 * 25 * 16);
  std::mt19937 rng2(42);
  std::uniform_int_distribution<int> dist2(0, std::numeric_limits<int>::max());
  for (size_t i = 0; i < n * c * h * w; i++) {
    a1[i] = dist2(rng2);
  }
  for (size_t i = 0; i < 3 * 25 * 16; i++) {
    a2[i] = dist2(rng2);
  }
  Tensor input = make_tensor(a1, test_shape);
  Tensor kernel = make_tensor(a2, Shape({5, 5, 3, 16}));
  Tensor output;

  RuntimeOptions options_seq;
  options_seq.par_backend = ParBackend::kSeq;

  RuntimeOptions options_stl;
  options_stl.par_backend = ParBackend::kTbb;

  ConvolutionalLayer p1(1, 1, 2, kernel);
  ConvolutionalLayer p2(1, 1, 2, kernel);
  double count1 = elapsed_time<double, std::milli>(test_func, p1, input, output,
                                                   options_seq);
  double count2 = elapsed_time<double, std::milli>(test_func, p2, input, output,
                                                   options_stl);
  std::cout << count1 << " vs. " << count2 << " (parallel)\n";

#ifdef ITLABAI_HAS_SYCL
  RuntimeOptions options_sycl;
  options_sycl.par_backend = ParBackend::kSycl;
  ConvolutionalLayer p3(1, 1, 2, kernel);
  double count3 = elapsed_time<double, std::milli>(test_func, p3, input, output,
                                                   options_sycl);
  std::cout << count1 << " vs. " << count3 << " (sycl)\n";
#endif
}
