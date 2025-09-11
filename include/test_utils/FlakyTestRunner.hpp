#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace test_utils {

static int getFlakyRetries() {
  const char* retry_env = std::getenv("FLAKY_RETRIES");
  if (retry_env != nullptr) {
    try {
      int retries = std::stoi(retry_env);
      return std::max(1, std::min(retries, 100));
    } catch (const std::exception&) {
      return 10;
    }
  }
  return 10;
}

static bool isFlakyTestsEnabled() {
  const char* disabled = std::getenv("DISABLE_FLAKY_TESTS");
  return !(disabled != nullptr && (std::strcmp(disabled, "1") == 0 ||
                                   std::strcmp(disabled, "true") == 0));
}

template <typename TestFunc>
void runFlakyTest(const char* test_name, TestFunc test_func) {
  if (!isFlakyTestsEnabled()) {
    test_func();
    return;
  }

  int max_retries = getFlakyRetries();

  for (int attempt = 1; attempt <= max_retries; ++attempt) {
    try {
      if (attempt > 1) {
        std::cout << "[FLAKY RETRY " << attempt << "/" << max_retries << "] "
                  << test_name << std::endl;
      }

      test_func();

      if (attempt > 1) {
        std::cout << "[FLAKY SUCCESS] " << test_name << " passed on attempt "
                  << attempt << std::endl;
      }
      return;

    } catch (const ::testing::AssertionException&) {
      if (attempt == max_retries) {
        std::cout << "[FLAKY EXHAUSTED] " << test_name << " failed after "
                  << max_retries << " attempts" << std::endl;
        throw;
      } else if (attempt == 1) {
        std::cout << "[FLAKY FAILED] " << test_name << " failed on attempt "
                  << attempt << ", retrying..." << std::endl;
      }
    }
  }
}

#define FLAKY_TEST(test_case_name, test_name) \
  TEST(test_case_name, test_name) {           \
        test_utils::runFlakyTest(#test_case_name "." #test_name, []() {
#define FLAKY_TEST_F(test_fixture, test_name) \
  TEST_F(test_fixture, test_name) {           \
        test_utils::runFlakyTest(#test_fixture "." #test_name, [this]() {
#define FLAKY_END_TEST \
  });            \
  }

}
