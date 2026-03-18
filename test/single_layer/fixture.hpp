#pragma once
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

using it_lab_ai::Backend;
using it_lab_ai::ParBackend;
using it_lab_ai::RuntimeOptions;
using it_lab_ai::Shape;

class BaseTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    defaultOptions.backend = Backend::kNaive;
    defaultOptions.par_backend = ParBackend::kSeq;
  }

  static RuntimeOptions setTBBOptions() {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = ParBackend::kTbb;
    return options;
  }

  static RuntimeOptions setSeqOptions() {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = ParBackend::kSeq;
    return options;
  }

  static RuntimeOptions setSTLOptions() {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = ParBackend::kThreads;
    return options;
  }

  static RuntimeOptions setKokkosOptions() {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = ParBackend::kKokkos;
    return options;
  }

  static RuntimeOptions setOmpOptions() {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = ParBackend::kOmp;
    return options;
  }

  static RuntimeOptions createOptionsWithBackend(ParBackend backend) {
    RuntimeOptions options;
    options.backend = Backend::kNaive;
    options.par_backend = backend;
    return options;
  }

  static std::vector<float> basic1DData() {
    return {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
  }

  static Shape basic1DShape() {
    return {8};
  }

  static std::vector<float> basic2DData4x4() {
    return {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f,
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  }

  static Shape basic2DShape4x4() {
    return {4, 4};
  }

  static std::vector<float> basic2DData3x3() {
    return {9.0f, 8.0f, 7.0f, 5.0f, 4.0f, 3.0f, 2.0f, 3.0f, 4.0f};
  }

  static Shape basic2DShape3x3() {
    return {3, 3};
  }

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

  static std::vector<float> ascending1DData() {
    return {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  }

  static Shape ascending1DShape() {
    return {10};
  }

  static std::vector<float> descending1DData() {
    return {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  }

  static std::vector<float> mixed1DData() {
    return {-5.0f, -3.0f, 0.0f, 2.0f, 4.0f, -1.0f, 3.0f, 1.0f, -2.0f, 5.0f};
  }

  static std::vector<float> small2DData2x2() {
    return {1.0f, 2.0f, 3.0f, 4.0f};
  }

  static Shape small2DShape2x2() {
    return {2, 2};
  }

  static std::vector<float> medium2DData5x5() {
    return {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f};
  }

  static Shape medium2DShape5x5() {
    return {5, 5};
  }

  static std::vector<float> zero2DData3x3() {
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  }

  static Shape zero2DShape3x3() {
    return {3, 3};
  }

  static std::vector<float> constant2DData4x4(float value = 5.0f) {
    return std::vector<float>(16, value);
  }

  static Shape constant2DShape4x4() {
    return {4, 4};
  }

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
};
