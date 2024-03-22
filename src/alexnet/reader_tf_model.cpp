#include "alexnet/reader_tf_model.hpp"

#include <fstream>
#include <iostream>
#include <vector>

void CheckStatus(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Error: " << TF_Message(status)
              << std::endl;
    TF_DeleteStatus(status);
    exit(1);
  }
}

Graph readTFModel(const std::string& modelPath, Graph& g) {

  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_SessionOptions* sessionOpts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sessionOpts, status);
  CheckStatus(status);

  std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Error: Failed to open TensorFlow model file." << std::endl;
    exit(1);
  }
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    std::cerr << "Error: Failed to read TensorFlow model file." << std::endl;
    exit(1);
  }
  file.close();

  TF_Buffer graphDef = {buffer.data(), static_cast<size_t>(fileSize), nullptr};
  TF_ImportGraphDefOptions* importOpts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, &graphDef, importOpts, status);
  CheckStatus(status);

  TF_DeleteImportGraphDefOptions(importOpts);

  TF_DeleteBuffer(&graphDef);

  TF_Operation* op;
  size_t pos1 = 0;
  std::vector<int> def;
  while ((op = TF_GraphNextOperation(graph, &pos1)) != nullptr) {
    std::string name = TF_OperationName(op);
    LayerType type;
    if (name.find("input") != std::string::npos) {
      type = kInput;
    } else if (name.find("pooling") != std::string::npos) {
      type = kPooling;
    } else if (name.find("normalization") != std::string::npos) {
      type = kNormalization;
    } else if (name.find("dropout") != std::string::npos) {
      type = kDropout;
    } else if (name.find("element_wise") != std::string::npos) {
      type = kElementWise;
    } else if (name.find("convolution") != std::string::npos) {
      type = kConvolution;
    } else if (name.find("fully_connected") != std::string::npos) {
      type = kFullyConnected;
    } else if (name.find("output") != std::string::npos) {
      type = kOutput;
    } else {
      throw std::runtime_error("Unknown node type: " + name);
    }
    LayerExample layer(type);
    g.setInput(layer, def);
    std::cout << "Added layer: " << name << " of type: " << type << std::endl;
  }

  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sessionOpts);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);

  return g;
}
