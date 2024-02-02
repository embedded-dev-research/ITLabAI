#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "perf/benchmarking.hpp"

// ==========================
// Timer tests

void waitfor_function(const size_t ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// chrono
TEST(timer, is_elapsed_time_returns_positive_value) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}
TEST(timer, is_elapsed_time_avg_returns_positive_value) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}

// omp
TEST(timer, is_elapsed_time_omp_returns_positive_value) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time_omp(waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}
TEST(timer, is_elapsed_time_omp_avg_returns_positive_value) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_omp_avg(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}

// >= -100 ms, <= +1000 ms
// chrono
TEST(timer, is_elapsed_time_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 1250);
}
TEST(timer, is_elapsed_time_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 1250);
}

// omp
TEST(timer, is_elapsed_time_omp_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time_omp(waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 1.25);
}
TEST(timer, is_elapsed_time_omp_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_omp_avg(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 1.25);
}

// ==========================
// Accuracy tests

TEST(accuracy, max_accuracy_test) {
  double a[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double b[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  auto acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 0.0, 1e-5);
}

TEST(accuracy, bad_accuracy_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  auto acc = accuracy<double>(a, b, 2);
  EXPECT_NEAR(acc, 99.3, 1e-5);
}

TEST(accuracy, bad_accuracy_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  auto acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 122.27, 1e-5);
}

TEST(accuracy, bad_accuracy_test_L) {
  size_t n = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < n; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < n; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < n; i++) {
    actual_acc += std::abs(a[i] - b[i]);
  }
  auto acc = accuracy<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  auto acc = accuracy_norm<double>(a, b, 2);
  EXPECT_NEAR(acc, 98.10734, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  auto acc = accuracy_norm<double>(a, b, 10);
  EXPECT_NEAR(acc, 52.72274, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_L) {
  size_t n = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < n; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < n; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < n; i++) {
    actual_acc += (a[i] - b[i]) * (a[i] - b[i]);
  }
  actual_acc = std::sqrt(actual_acc);
  auto acc = accuracy_norm<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, accuracy_throws_when_bad_pointer) {
  double *a = nullptr;
  auto *b = new double[5];
  EXPECT_ANY_THROW(accuracy<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy<double>(b, a, 5));
  delete[] b;
}

TEST(accuracy, accuracy_norm_throws_when_bad_pointer) {
  double *a = nullptr;
  auto *b = new double[5];
  EXPECT_ANY_THROW(accuracy_norm<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy_norm<double>(b, a, 5));
  delete[] b;
}

// ==========================
// Throughput tests

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
      a[ptr] = i + j;
      b[ptr] = i - j;
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
      a[ptr] = i + j;
      b[ptr] = i - j;
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
      a[ptr] = i + j;
      b[ptr] = i - j;
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
      a[ptr] = i + j;
      b[ptr] = i - j;
      ptr++;
    }
  }
  double tp;
  tp = throughput_omp_avg(10, matrix_sum<int>, a, b);
  EXPECT_GE(tp, 0);
  tp = throughput_omp_avg(10, matrix_mul<int>, n, a, b);
  EXPECT_GE(tp, 0);
}
