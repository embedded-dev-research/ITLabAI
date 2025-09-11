#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  const char* flaky_disabled = std::getenv("DISABLE_FLAKY_TESTS");
  bool flaky_enabled =
      (flaky_disabled == nullptr) || (std::strcmp(flaky_disabled, "1") != 0 &&
                                      std::strcmp(flaky_disabled, "true") != 0);

  if (flaky_enabled) {
    const char* retries = std::getenv("FLAKY_RETRIES");
    int retry_count = retries ? std::atoi(retries) : 10;
    std::cout << "Flaky test support enabled. Max retries: " << retry_count
              << std::endl;
    std::cout << "Use FLAKY_TEST/FLAKY_TEST_F macros to create flaky tests."
              << std::endl;
  }

  return RUN_ALL_TESTS();
}
