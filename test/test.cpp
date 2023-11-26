#include "gtest/gtest.h"
#include "timer.hpp"

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
