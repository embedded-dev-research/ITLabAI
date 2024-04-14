#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

TEST(input, chech_basic) {
  Shape sh1({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(0, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  ASSERT_EQ(tmp.size(), 4);
}
TEST(input, run_int) {
  Shape sh1({2, 2});
  std::vector<int> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(0, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_float) {
  Shape sh1({2, 2});
  std::vector<float> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(0, 1, 2);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}