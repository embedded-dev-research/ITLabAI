#include <stdexcept>

#include "gtest/gtest.h"
#include "parallel/parallel.hpp"

#ifdef ITLABAI_HAS_SYCL
#  include <vector>
#endif

namespace it_lab_ai {
namespace {

#ifdef ITLABAI_HAS_SYCL
TEST(sycl_parallel, parallel_for_writes_expected_values) {
  constexpr std::size_t kSize = 1024;
  std::vector<int> values(kSize, 0);

  parallel::parallel_for(kSize, [&](std::size_t i) {
    values[i] = static_cast<int>(i * 2);
  }, parallel::Backend::kSycl);

  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], static_cast<int>(i * 2));
  }
}

TEST(sycl_parallel, reports_selected_device) {
  EXPECT_FALSE(parallel::sycl_device_name().empty());
}

#else

TEST(sycl_parallel, throws_when_backend_is_unavailable) {
  EXPECT_THROW(parallel::parallel_for(1024, [](std::size_t) {},
                                      parallel::Backend::kSycl),
               std::runtime_error);
}

#endif

}  // namespace
}  // namespace it_lab_ai
