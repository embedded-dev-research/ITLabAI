#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

TEST(input, chech_basic) {
  const std::string image_path = IMAGE_PATH;
  int n = 1;
  std::vector<std::string> paths;
  paths.push_back(image_path);
  InputLayer<int> inlayer(n);
  Tensor output = inlayer.run(paths);
  std::vector<int> tmp = *output.as<int>();
  ASSERT_EQ(tmp.size(), 227 * 227 * 3);
}
