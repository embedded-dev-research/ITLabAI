#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

TEST(input, chech_basic) {
  const std::string image_path = IMAGE_PATH;
  int n = 1;
  std::vector<std::string> paths;
  paths.push_back(image_path);
  Shape sh1({2, 2});
  std::vector<int> vec = {1, 2, 3, 4};
  Tensor output = make_tensor(vec, sh1);
  InputLayer inlayer;
  inlayer.run(paths, output);
  std::vector<int> tmp = *output.as<int>();
  ASSERT_EQ(tmp.size(), 227 * 227 * 3);
}
