#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "alexnet/alexnet.hpp"
#include "gtest/gtest.h"

TEST(Read_model, can_make_session) {
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* session_options = TF_NewSessionOptions();
  ASSERT_NO_THROW(TF_NewSession(graph, session_options, status););
}
TEST(Read_model, can_read_model) {
  std::string path = MODEL_PATH;
  std::ifstream file(path, std::ios::binary);

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);

  ASSERT_NO_THROW(file.read(buffer.data(), file_size););
  file.close();
}
