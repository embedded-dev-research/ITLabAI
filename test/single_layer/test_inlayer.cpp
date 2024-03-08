#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

TEST(input, chech_basic) {
  const std::string image_path = IMAGE_PATH;
  int n = 1;
  InputLayer<int> inlayer(n);
  std::vector<int> output = inlayer.run(image_path);
  ASSERT_EQ(output.size(), 227 * 227 * 3);
}
