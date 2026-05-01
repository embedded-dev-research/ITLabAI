#include "gtest/gtest.h"
#include "parallel/parallel.hpp"

#ifdef ITLABAI_HAS_SYCL

#include <vector>

namespace it_lab_ai {
namespace {

TEST(sycl_parallel, parallel_for_writes_expected_values) {
  constexpr std::size_t kSize = 1024;
  std::vector<int> values(kSize, 0);

  parallel::parallel_for(
      kSize,
      [&](std::size_t i) {
        values[i] = static_cast<int>(i * 2);
      },
      parallel::Backend::kSycl);

  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], static_cast<int>(i * 2));
  }
}

TEST(sycl_parallel, reports_selected_device) {
  EXPECT_FALSE(parallel::sycl_device_name().empty());
}

}  // namespace
}  // namespace it_lab_ai

#endif
