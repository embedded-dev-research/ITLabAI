#include <thread>

#include "benchmarking.hpp"
#include "gtest/gtest.h"

TEST(basic, basic_test) {
  // Arrange
  int a = 2;
  int b = 3;

  // Act
  int c = a + b;

  // Assert
  ASSERT_EQ(5, c);
}

// ==========================
// Timer tests

void waitfor_function(const size_t ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

TEST(timer, is_elapsed_time_returns_positive_value) {
  const size_t a = 500;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}
TEST(timer, is_elapsed_time_avg_returns_positive_value) {
  const size_t a = 500;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.0);
}

// +- 100 ms
TEST(timer, is_elapsed_time_returns_nearly_correct_time) {
  const size_t a = 500;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 400);
  EXPECT_LE(res_time, 600);
}
TEST(timer, is_elapsed_time_avg_returns_nearly_correct_time) {
  const size_t a = 500;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 400);
  EXPECT_LE(res_time, 600);
}

// ==========================

// ==========================
// Accuracy tests

TEST(accuracy, max_accuracy_test) {
  double a[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double b[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double acc = accuracy<double>(a, b, 10);
  EXPECT_EQ(acc, 0.0);
}

TEST(accuracy, bad_accuracy_test) {
  double a[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double b[10] = {0.0, -6.0, 12.0, 44.0, -7.0, 11.0, 12.0, 0.0, 0.0, -7.0};
  double acc = accuracy<double>(a, b, 10);
  EXPECT_EQ(acc, 122.0);
}

TEST(accuracy, throws_when_bad_pointer) {
  double *a = nullptr;
  double *b = new double[5];
  EXPECT_ANY_THROW(accuracy<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy<double>(b, a, 5));
  delete[] b;
}

// ==========================
