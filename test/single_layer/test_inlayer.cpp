#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

using namespace itlab_2023;

TEST(input, chech_basic) {
  Shape sh1({1, 2, 2, 1});
  std::vector<int> vec = {1, 2, 3, 4};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNhwc, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  ASSERT_EQ(tmp.size(), 4);
}
TEST(input, run_int) {
  Shape sh1({1, 2, 2, 1});
  std::vector<int> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNhwc, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_float) {
  Shape sh1({1, 2, 2, 1});
  std::vector<float> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNhwc, 1, 2);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_int_NCHW_NHWC) {
  Shape sh1({1, 1, 2, 2});
  std::vector<int> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNchw, kNhwc, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_float_NCHW_NHWC) {
  Shape sh1({1, 1, 2, 2});
  std::vector<float> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNchw, kNhwc, 1, 2);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_int_NHWC_NCHW) {
  Shape sh1({1, 2, 2, 1});
  std::vector<int> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNchw, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_float_NHWC_NCHW) {
  Shape sh1({1, 2, 2, 1});
  std::vector<float> vec = {3, 5, 7, 9};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNchw, 1, 2);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> res = {1, 2, 3, 4};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_int_NHWC_NCHW_C3) {
  Shape sh1({1, 2, 2, 3});
  std::vector<int> vec = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNchw, 1, 2);
  layer.run(input, output);
  std::vector<int> tmp = *output.as<int>();
  std::vector<int> res = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
  ASSERT_EQ(tmp, res);
}
TEST(input, run_float_NHWC_NCHW_C3) {
  Shape sh1({1, 2, 2, 3});
  std::vector<float> vec = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
  Tensor input = make_tensor(vec, sh1);
  Tensor output = make_tensor(vec, sh1);
  InputLayer layer(kNhwc, kNchw, 1, 2);
  layer.run(input, output);
  std::vector<float> tmp = *output.as<float>();
  std::vector<float> res = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
  ASSERT_EQ(tmp, res);
}
