#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

TEST(input, chech_basic) {
  char* workspace = getenv("GITHUB_WORKSPACE");
  std::string input_file_name =
      std::string(workspace) + "/build/test/image.jpg";
  int n = 1;
  InputLayer<int> inlayer(n);
  std::vector<int> output = inlayer.run(input_file_name);
  ASSERT_EQ(output.size(), 227 * 227 * 3);
}
