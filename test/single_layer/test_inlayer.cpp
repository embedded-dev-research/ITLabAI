#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "layers/InputLayer.hpp"

std::string getPath() {
  char* workspace = getenv("GITHUB_WORKSPACE");
  if (workspace != nullptr) {
    return std::string(workspace) + "/build/test/image.jpg";
  }
  return "../image.jpg";
}

TEST(input, chech_basic) {
  std::string input_file_name = getPath();
  int N = 1;
  InputLayer<int> inlayer(N);
  std::vector<int> output = inlayer.run(input_file_name);
  ASSERT_EQ(output.size(), 227 * 227 * 3);
}
