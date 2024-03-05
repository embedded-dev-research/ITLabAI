#include <tensorflow/c/c_api.h>

#include <fstream>
#include <iostream>
#include <vector>
void AlexNetSample(std::string& path) {
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* session_options = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sessionOptions, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Error creating TensorFlow session: " << TF_Message(status)
              << std::endl;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening model file: " << path << std::endl;
  }

  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fileSize);
  file.read(buffer.data(), fileSize);
  file.close();

  TF_Buffer* modelBuffer = TF_NewBufferFromString(buffer.data(), buffer.size());
  TF_ImportGraphDefOptions* importOptions = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, modelBuffer, importOptions, status);
  TF_DeleteImportGraphDefOptions(importOptions);
  TF_DeleteBuffer(modelBuffer);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Error importing graph definition: " << TF_Message(status)
              << std::endl;
  }

  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sessionOptions);
  TF_DeleteStatus(status);
}
