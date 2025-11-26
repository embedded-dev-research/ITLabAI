#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "graph_transformations/graph_transformations.hpp"
#include "perf/benchmarking.hpp"
#include "build.cpp"
#include "build.hpp"

using namespace it_lab_ai;

int main() {
  Tensor aaaa = make_tensor(std::vector<int>({0}));
  auto graph1 = build_graph(aaaa, aaaa, MODEL_PATH_DENSENET_ONNX, false);

  Graph subgraph(5);
  Tensor scale = make_tensor(std::vector<float>({1.0}));
  std::shared_ptr<Layer> layer_0 =
      std::make_shared<BatchNormalizationLayer>(scale, scale, scale, scale);
  std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
  std::shared_ptr<Layer> layer_2 = std::make_shared<ConvolutionalLayer>();
  std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
  std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
  subgraph.setInput(layer_0, aaaa);
  subgraph.makeConnection(layer_0, layer_1);
  subgraph.makeConnection(layer_1, layer_2);
  subgraph.makeConnection(layer_2, layer_3);
  subgraph.makeConnection(layer_3, layer_4);

  Graph subgraph2(5);
  std::shared_ptr<Layer> layer_5 = std::make_shared<ConcatLayer>();
  std::shared_ptr<Layer> layer_6 =
      std::make_shared<PoolingLayer>(Shape({1, 1, 1}), "max");
  std::shared_ptr<Layer> layer_7 = std::make_shared<ConvolutionalLayer>();
  subgraph2.setInput(layer_6, aaaa);
  subgraph2.makeConnection(layer_6, layer_5);
  subgraph2.addSingleLayer(layer_7);
  subgraph2.makeConnection(layer_7, layer_5);

  auto vec = find_subgraphs(graph1, subgraph);
  auto vec2 = find_subgraphs(graph1, subgraph2);
  auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph1,
                                                   subgraph);
  auto time2 = elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph1,
                                                    subgraph2);
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < vec[i].size(); j++) {
      std::cerr << vec[i][j] << ' ';
    }
    std::cerr << '\n';
  }
  std::cerr << "Time for path5:" << time << std::endl;

  for (int i = 0; i < vec2.size(); i++) {
    for (int j = 0; j < vec2[i].size(); j++) {
      std::cerr << vec2[i][j] << ' ';
    }
    std::cerr << '\n';
  }
  std::cerr << "Time for concat:" << time2 << std::endl;
  return 0;
}
