#include <cstdlib>
#include <fstream>
#include <iostream>

#include "gtest/gtest.h"
#include "layers/OutputLayer.hpp"

TEST(OutputLayer, can_get_topk) {
  const int k = 50;
  std::ifstream f;
  f.open(std::string(TESTS_BINARY_PATH) + "/imagenet-labels.txt", std::ios::in);
  std::vector<std::string> labels;
  std::vector<double> input;
  char* buf = new char[257];
  if (f.fail()) {
    throw std::runtime_error("No such file");
  }
  // get labels
  while (!f.eof()) {
    f.getline(buf, 256);
    labels.emplace_back(buf);
  }
  delete[] buf;
  // get random nums
  for (size_t i = 0; i < labels.size(); i++) {
    input.push_back(static_cast<double>(std::rand()) / RAND_MAX);
  }
  OutputLayer<double> a({labels.size()}, labels);
  // debug
  auto topk = a.top_k(input, k);
  for (size_t i = 0; i < topk.first.size(); i++) {
    std::cerr << i + 1 << ". " << topk.first[i] << ' ' << topk.second[i]
              << std::endl;
  }
  ASSERT_NO_THROW(auto topk1 = a.top_k(input, k));
}
