#include <fstream>
#include <iostream>
#include <vector>

#include "alexnet/alexnet.hpp"

void AlexNetSample(std::string& path) {
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* session_options = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, session_options, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Error creating TensorFlow session: " << TF_Message(status)
              << std::endl;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening model file: " << path << std::endl;
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  file.read(buffer.data(), file_size);
  file.close();
}