#include <fstream>
#include <iostream>
#include <vector>

#include "alexnet.hpp"

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

  TF_Buffer* model_buffer =
      TF_NewBufferFromString(buffer.data(), buffer.size());
  TF_ImportGraphDefOptions* import_options = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, model_buffer, import_options, status);
  TF_DeleteImportGraphDefOptions(import_options);
  TF_DeleteBuffer(model_buffer);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Error importing graph definition: " << TF_Message(status)
              << std::endl;
  }

  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(session_options);
  TF_DeleteStatus(status);
}
