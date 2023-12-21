#include <random>
#include <thread>

#include "Graph/Graph.hpp"
#include "gtest/gtest.h"
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
TEST(graph, check_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_connection1) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_connection_when_not_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_connection_when_not_connection2) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_connection_when_not_connection3) {
  const std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}

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
