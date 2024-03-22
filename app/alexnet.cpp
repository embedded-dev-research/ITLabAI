#include "alexnet/alexnet.hpp"
#include <iostream>

#include "alexnet/reader_tf_model.hpp"
int main() {
  std::string model_path = MODEL1_PATH;
  //AlexNetSample(model_path);
  Graph graph(3);
  readTFModel(model_path, graph);

}
