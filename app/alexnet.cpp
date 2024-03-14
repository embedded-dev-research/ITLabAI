#include "alexnet/alexnet.hpp"
#include <iostream>

#include "alexnet/reader_tf_model.hpp"
int main() {
  std::string image_path = MODEL1_PATH;
  AlexNetSample(image_path);
  Graph graph(3);
  graph = readTFModel(image_path);
}
