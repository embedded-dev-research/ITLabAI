#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "perf/benchmarking.hpp"

template <typename T>
std::vector<T> matrix_sum(const std::vector<T> &first,
                          const std::vector<T> &second) {
  std::vector<T> res(first.size());
  std::transform(first.begin(), first.end(), second.begin(), res.begin(),
                 std::plus<T>());
  return res;
}

template <typename T>
std::vector<T> matrix_mul(const size_t n, const std::vector<T> &first,
                          const std::vector<T> &second) {
  std::vector<T> mul(n * n, T(0));
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < n; k++) {
      for (size_t j = 0; j < n; j++) {
        mul[n * i + j] += first[n * i + k] * second[n * k + j];
      }
    }
  }
  return mul;
}

TEST(throughput, matrix_operations_throughput_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = static_cast<int>(i + j);
      b[ptr] = static_cast<int>(i - j);
      ptr++;
    }
  }
  double tp;
  tp = throughput<double, std::ratio<1, 1> >(matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput<double, std::ratio<1, 1> >(matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_avg_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = static_cast<int>(i + j);
      b[ptr] = static_cast<int>(i - j);
      ptr++;
    }
  }
  double tp;
  tp = throughput_avg<double, std::ratio<1, 1> >(10, matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_avg<double, std::ratio<1, 1> >(10, matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_omp_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = static_cast<int>(i + j);
      b[ptr] = static_cast<int>(i - j);
      ptr++;
    }
  }
  double tp;
  tp = throughput_omp(matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_omp(matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
TEST(throughput, matrix_operations_throughput_omp_avg_is_positive) {
  size_t n = 200;
  std::vector<int> a(n * n);
  std::vector<int> b(n * n);
  size_t ptr = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a[ptr] = static_cast<int>(i + j);
      b[ptr] = static_cast<int>(i - j);
      ptr++;
    }
  }
  double tp;
  tp = throughput_omp_avg(10, matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_omp_avg(10, matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
