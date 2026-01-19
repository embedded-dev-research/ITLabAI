#pragma once
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

class BaseTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    defaultOptions.backend = Backend::kNaive;
    defaultOptions.parallel = false;
    defaultOptions.par_backend = ParBackend::kSeq;
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

 protected:
  RuntimeOptions defaultOptions;
};
