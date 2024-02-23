#include <tensorflow/c/c_api.h>
#include <fstream>
#include <iostream>
#include <vector> 
void AlexNetSample(std::string& path) {
  // Initialize TensorFlow session
TF_Graph* graph = TF_NewGraph();
TF_Status* status = TF_NewStatus();
TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
TF_Session* session = TF_NewSession(graph, sessionOptions, status);

if (TF_GetCode(status) != TF_OK) {
std::cerr << "Error creating TensorFlow session: " << TF_Message(status) << std::endl;
}

// Read model file
std::ifstream file(path, std::ios::binary);
if (!file.is_open()) {
std::cerr << "Error opening model file: " << path << std::endl;
}

// Determine file size
file.seekg(0, std::ios::end);
size_t fileSize = file.tellg();
file.seekg(0, std::ios::beg);

// Read file content into buffer
std::vector<char> buffer(fileSize);
file.read(buffer.data(), fileSize);
file.close();

// Load model into graph
TF_Buffer* modelBuffer = TF_NewBufferFromString(buffer.data(), buffer.size());
TF_ImportGraphDefOptions* importOptions = TF_NewImportGraphDefOptions();
TF_GraphImportGraphDef(graph, modelBuffer, importOptions, status);
TF_DeleteImportGraphDefOptions(importOptions);
TF_DeleteBuffer(modelBuffer);

if (TF_GetCode(status) != TF_OK) {
std::cerr << "Error importing graph definition: " << TF_Message(status) << std::endl;
}

// Perform inference (replace this with your inference code)
// ...

// Clean up
TF_DeleteGraph(graph);
TF_DeleteSession(session, status);
TF_DeleteSessionOptions(sessionOptions);
TF_DeleteStatus(status);
}


