#pragma once
#include <gtest/gtest.h>

#include <algorithm>  // для std::replace
#include <random>
#include <string>  // добавьте это
#include <vector>

#include "layers/ConvLayer.hpp"

// ТОЛЬКО ПОСЛЕ включения всех заголовков
using namespace it_lab_ai;

class BaseTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    defaultOptions.backend = Backend::kNaive;
    defaultOptions.parallel = false;
    defaultOptions.par_backend = ParBackend::kSeq;

    rng.seed(42);
  }

  RuntimeOptions setTBBOptions() const {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.parallel = true;
    options.par_backend = ParBackend::kTbb;
    return options;
  }

  RuntimeOptions setSeqOptions() const {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.parallel = true;
    options.par_backend = ParBackend::kSeq;
    return options;
  }

  RuntimeOptions setSTLOptions() const {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.parallel = true;
    options.par_backend = ParBackend::kThreads;
    return options;
  }

  RuntimeOptions createOptionsWithBackend(ParBackend backend) const {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.parallel = (backend != ParBackend::kSeq);
    options.par_backend = backend;
    return options;
  }

 public:
  static std::vector<float> basic1DData() {
    return {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
  }

  static Shape basic1DShape() { return {8}; }

  static std::vector<float> basic2DData4x4() {
    return {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f,
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  }

  static Shape basic2DShape4x4() { return {4, 4}; }

  static std::vector<float> basic2DData3x3() {
    return {9.0f, 8.0f, 7.0f, 5.0f, 4.0f, 3.0f, 2.0f, 3.0f, 4.0f};
  }

  static Shape basic2DShape3x3() { return {3, 3}; }

  static std::vector<float> activationTestData() {
    return {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  }

  static std::vector<float> reluExpected() {
    return {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  }

  static std::vector<float> sigmoidExpected() {
    return {0.0474f, 0.1192f, 0.2689f, 0.5f, 0.7311f, 0.8808f, 0.9526f};
  }

  static std::vector<float> get1DAverageExpected() {
    return {8.0f, 6.0f, 4.0f};
  }

  static std::vector<float> get2DAverageStride1Expected() {
    return {6.5f, 5.5f, 4.5f, 3.5f, 3.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  }

  template <typename T>
  std::vector<T> generateRandomVector(size_t size, T min, T max) {
    std::uniform_real_distribution<T> dist(min, max);
    std::vector<T> result(size);
    for (size_t i = 0; i < size; ++i) {
      result[i] = dist(rng);
    }
    return result;
  }

  template <typename T>
  Tensor generateRandomTensor(const Shape& shape, T min, T max) {
    auto data = generateRandomVector<T>(shape.count(), min, max);
    return make_tensor(data, shape);
  }

  void resetRNG() { rng.seed(42); }

  template <typename T>
  static void expectVectorsNear(const std::vector<T>& actual,
                                const std::vector<T>& expected,
                                T tolerance = static_cast<T>(1e-5)) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
    }
  }

 protected:
  RuntimeOptions defaultOptions;
  mutable std::mt19937 rng;

  static constexpr size_t SMALL_SIZE = 1000;
  static constexpr size_t MEDIUM_SIZE = 10000;
  static constexpr size_t LARGE_SIZE = 100000;
  static constexpr size_t PERFORMANCE_TEST_SIZE = 8000000;
};
