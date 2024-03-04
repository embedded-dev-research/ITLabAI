#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

std::string getPath() {
  char* workspace = getenv("GITHUB_WORKSPACE");
  return std::string(workspace) + "/build/test/image.jpg";
}

TEST(input, chech_basic) {
  std::string input_file_name = getPath();
  int n = 1;
  InputLayer<int> inlayer(n);
  std::vector<int> output = inlayer.run(input_file_name);
  ASSERT_EQ(output.size(), 227 * 227 * 3);
}
