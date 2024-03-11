#include <fstream>
#include <iostream>
#include <vector>

#include "alexnet/reader_tf_model.hpp"

Graph readTFModel(const std::string& modelPath) {
  Graph graph(3);

  // Чтение содержимого файла
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + modelPath);
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  file.read(buffer.data(), file_size);
  file.close();

  // Загружаем модель TensorFlow из буфера
  TF_Graph* graphDef = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_Buffer* tf_buffer = TF_NewBufferFromString(buffer.data(), buffer.size());

  // Создаем сессию
  TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graphDef, sessionOptions, status);
  if (TF_GetCode(status) != TF_OK) {
    throw std::runtime_error("Failed to create TensorFlow session: " +
                             std::string(TF_Message(status)));
  }

  // Импортируем граф из буфера
  TF_ImportGraphDefOptions* importOptions = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graphDef, tf_buffer, importOptions, status);
  TF_DeleteImportGraphDefOptions(importOptions);
  TF_DeleteBuffer(tf_buffer);

  if (TF_GetCode(status) != TF_OK) {
    throw std::runtime_error("Failed to import graph definition: " +
                             std::string(TF_Message(status)));
  }

  TF_Operation* op;
  size_t pos = 0;
  while ((op = TF_GraphNextOperation(graphDef, &pos)) != nullptr) {
    std::string name = TF_OperationName(op);
    // Определяем тип узла на основе его имени или атрибутов
    LayerType type;
    // Определяем тип узла на основе его имени или атрибутов
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
      // Неизвестный тип узла
      throw std::runtime_error("Unknown node type: " + name);
    }
    // Создаем экземпляр узла и добавляем его в граф
    LayerExample layer(type);
    graph.setInput(layer, /* передать данные узла */);
  }

  // Освобождаем ресурсы
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sessionOptions);
  TF_DeleteStatus(status);
  TF_DeleteGraph(graphDef);

  // Возвращаем построенный граф
  return graph;
}
