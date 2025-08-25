#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "layers/ConvLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

void test_func(Layer& p, const Tensor& input, Tensor& output) {
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  p.run(in, out);
}

TEST(pooling_test, is_pooling_tbb_ok) {
  size_t n = 10;
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
  double count2 =
      elapsed_time<double, std::milli>(test_func, p2, input, output);
  std::cout << count1 << " vs. " << count2 << " (parallel)\n";
}

TEST(conv_test, is_conv_stl_ok) {
  size_t n = 10;
  size_t c = 3;
  size_t h = 224;
  size_t w = 224;
  Shape test_shape = {n, c, h, w};
  std::vector<int> a1(n * c * h * w);
  std::vector<int> a2(3 * 25 * 16);
  for (size_t i = 0; i < n * c * h * w; i++) {
    a1[i] = rand();
  }
  for (size_t i = 0; i < 3 * 25 * 16; i++) {
    a2[i] = rand();
  }
  Tensor input = make_tensor(a1, test_shape);
  Tensor kernel = make_tensor(a2, Shape({5, 5, 3, 16}));
  Tensor output;
  ConvolutionalLayer p1(1, 1, 2, kernel, Tensor(), kDefault);
  ConvolutionalLayer p2(1, 1, 2, kernel, Tensor(), kSTL);
  double count1 =
      elapsed_time<double, std::milli>(test_func, p1, input, output);
  double count2 =
      elapsed_time<double, std::milli>(test_func, p2, input, output);
  std::cout << count1 << " vs. " << count2 << " (parallel)\n";
}
