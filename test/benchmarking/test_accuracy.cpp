#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "perf/benchmarking.hpp"

using namespace itlab_2023;

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
