#include <random>
#include <thread>

#include "gtest/gtest.h"
#include "layers/FCLayer.hpp"
#include "perf/benchmarking.hpp"

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
// Fully Connected layer

TEST(fclayer, calculates_correctly1) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  std::vector<double> output = layer.run({1, 2});
  EXPECT_NEAR(output[0], 5.5, 1e-5);
  EXPECT_NEAR(output[1], 4.4, 1e-5);
  EXPECT_NEAR(output[2], 12.0, 1e-5);
}

TEST(fclayer, calculates_correctly2) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  std::vector<double> output = layer.run({0.5, 0.0});
  EXPECT_NEAR(output[0], 1.5, 1e-5);
  EXPECT_NEAR(output[1], 0.55, 1e-5);
  EXPECT_NEAR(output[2], 1.0, 1e-5);
}

TEST(fclayer, throws_when_greater_input_size) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> input = {2.0, 1.0, 0.0};
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_less_input_size) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> input = {2.0};
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}

TEST(fclayer, throws_when_empty_input) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> input;
  std::vector<double> bias = {0.5, 0.5, 1.0};
  FCLayer<double> layer(weights, bias);
  ASSERT_ANY_THROW(layer.run(input));
}
TEST(fclayer, throws_when_empty_bias) {
  const std::vector<double> a1 = {2.0, 1.5};
  const std::vector<double> a2 = {0.1, 1.9};
  const std::vector<double> a3 = {0.0, 5.5};
  std::vector<std::vector<double> > weights = {a1, a2, a3};
  std::vector<double> bias;
  ASSERT_ANY_THROW(FCLayer<double>(weights, bias));
}

// ==========================

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

// +- 100 ms
// chrono
TEST(timer, is_elapsed_time_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time<double, std::milli>(waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 350);
}
TEST(timer, is_elapsed_time_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_avg<double, std::milli>(b, waitfor_function, a);
  EXPECT_GE(res_time, 150);
  EXPECT_LE(res_time, 350);
}

// omp
TEST(timer, is_elapsed_time_omp_returns_nearly_correct_time) {
  const size_t a = 250;
  double res_time;
  res_time = elapsed_time_omp(waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 0.35);
}
TEST(timer, is_elapsed_time_omp_avg_returns_nearly_correct_time) {
  const size_t a = 250;
  const size_t b = 10;
  double res_time;
  res_time = elapsed_time_omp_avg(b, waitfor_function, a);
  EXPECT_GE(res_time, 0.15);
  EXPECT_LE(res_time, 0.35);
}

// ==========================

// ==========================
// Accuracy tests

TEST(accuracy, max_accuracy_test) {
  double a[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double b[10] = {9.0, 2.0, 1.0, 4.0, 7.0, 10.5, -12.0, 11.0, 0.0, -2.5};
  double acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 0.0, 1e-5);
}

TEST(accuracy, bad_accuracy_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  double acc = accuracy<double>(a, b, 2);
  EXPECT_NEAR(acc, 99.3, 1e-5);
}

TEST(accuracy, bad_accuracy_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  double acc = accuracy<double>(a, b, 10);
  EXPECT_NEAR(acc, 122.27, 1e-5);
}

TEST(accuracy, bad_accuracy_test_L) {
  size_t N = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < N; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < N; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < N; i++) {
    actual_acc += std::abs(a[i] - b[i]);
  }
  double acc = accuracy<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_S) {
  double a[2] = {0.5, 2.7};
  double b[2] = {1.7, 100.8};
  double acc = accuracy_norm<double>(a, b, 2);
  EXPECT_NEAR(acc, 98.10734, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_M) {
  double a[10] = {9.0,   2.5,   1.0,    4.0, 7.0,
                  10.48, -12.0, 10.494, 0.0, -2.240001};
  double b[10] = {0.0,  -6.0, 12.0, 44.006, -7.0,
                  11.0, 12.0, 0.0,  0.0,    -6.990001};
  double acc = accuracy_norm<double>(a, b, 10);
  EXPECT_NEAR(acc, 52.72274, 1e-5);
}

TEST(accuracy, bad_accuracy_norm_test_L) {
  size_t N = 5000;
  double a[5000];
  double b[5000];
  for (size_t i = 0; i < N; i++) {
    a[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  for (size_t i = 0; i < N; i++) {
    b[i] = (static_cast<double>(rand()) / RAND_MAX - 1.0) * 100;  // [-100;100]
  }
  double actual_acc = 0.0;
  for (size_t i = 0; i < N; i++) {
    actual_acc += (a[i] - b[i]) * (a[i] - b[i]);
  }
  actual_acc = std::sqrt(actual_acc);
  double acc = accuracy_norm<double>(a, b, 5000);
  EXPECT_NEAR(acc, actual_acc, 1e-5);
}

TEST(accuracy, accuracy_throws_when_bad_pointer) {
  double *a = nullptr;
  double *b = new double[5];
  EXPECT_ANY_THROW(accuracy<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy<double>(b, a, 5));
  delete[] b;
}

TEST(accuracy, accuracy_norm_throws_when_bad_pointer) {
  double *a = nullptr;
  double *b = new double[5];
  EXPECT_ANY_THROW(accuracy_norm<double>(a, b, 5));
  EXPECT_ANY_THROW(accuracy_norm<double>(b, a, 5));
  delete[] b;
}

// ==========================
