#include <string>
#include <cstdlib>

#include "layers/InputLayer.hpp"
#include "gtest/gtest.h"

std::string getPath() {
  char* workspace = getenv("GITHUB_WORKSPACE");
  return std::string(workspace) + "/build/test/image.jpg";
}

TEST(input, chech_basic) {
  std::string input_file_name = "D:/proekts/itlab_2023/build/test/image.jpg";
  int N = 1;
  InputLayer<int> inlayer(N);
  std::vector<int> output = inlayer.run(input_file_name);
  ASSERT_EQ(output.size(), 227 * 227 * 3);
}
