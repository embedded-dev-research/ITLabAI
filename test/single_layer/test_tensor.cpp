#include <cstdint>
#include "gtest/gtest.h"
#include "layers/Tensor.hpp"
#include <vector>

TEST(Tensor, can_create_int_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, 2, 1, 6, 3};
  std::vector<uint8_t> bytes_vals(std::begin(vals_tensor), std::end(vals_tensor));
  ASSERT_NO_THROW(Tensor t(bytes_vals, sh, kInt));
}

TEST(Tensor, can_create_double_tensor) {
  Shape sh({2, 3});
  std::vector<double> vals_tensor = {4, 0, 2, 1, 6, 3};
  std::vector<uint8_t> bytes_vals(std::begin(vals_tensor), std::end(vals_tensor));
  ASSERT_NO_THROW(Tensor t(bytes_vals, sh, kDouble));
}

