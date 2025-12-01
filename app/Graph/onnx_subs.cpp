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
  Graph graph1;
  build_graph(graph1, aaaa, aaaa, MODEL_PATH_DENSENET_ONNX, false);

  Graph subgraph;
  Tensor scale = make_tensor(std::vector<float>({1.0}));
  std::unique_ptr<Layer> layer_0 =
      std::make_unique<BatchNormalizationLayer>(scale, scale, scale, scale);
  std::unique_ptr<Layer> layer_1 = std::make_unique<EWLayer>("relu");
  std::unique_ptr<Layer> layer_2 = std::make_unique<ConvolutionalLayer>();
  std::unique_ptr<Layer> layer_3 = std::make_unique<EWLayer>("relu");
  std::unique_ptr<Layer> layer_4 = std::make_unique<ConvolutionalLayer>();

  Layer* layer_0_ptr = layer_0.get();
  Layer* layer_1_ptr = layer_1.get();
  Layer* layer_2_ptr = layer_2.get();
  Layer* layer_3_ptr = layer_3.get();
  Layer* layer_4_ptr = layer_4.get();

  subgraph.setInput(layer_0_ptr, aaaa);
  subgraph.makeConnection(layer_0_ptr, layer_1_ptr);
  subgraph.makeConnection(layer_1_ptr, layer_2_ptr);
  subgraph.makeConnection(layer_2_ptr, layer_3_ptr);
  subgraph.makeConnection(layer_3_ptr, layer_4_ptr);

  Graph subgraph2;
  std::unique_ptr<Layer> layer_5 = std::make_unique<ConcatLayer>();
  std::unique_ptr<Layer> layer_6 =
      std::make_unique<PoolingLayer>(Shape({1, 1, 1}), "max");
  std::unique_ptr<Layer> layer_7 = std::make_unique<ConvolutionalLayer>();

  Layer* layer_5_ptr = layer_5.get();
  Layer* layer_6_ptr = layer_6.get();
  Layer* layer_7_ptr = layer_7.get();

  subgraph2.setInput(layer_6_ptr, aaaa);
  subgraph2.makeConnection(layer_6_ptr, layer_5_ptr);
  subgraph2.addSingleLayer(layer_7_ptr);
  subgraph2.makeConnection(layer_7_ptr, layer_5_ptr);

  std::vector<std::vector<int>> vec = find_subgraphs(graph1, subgraph);
  std::vector<std::vector<int>> vec2 = find_subgraphs(graph1, subgraph2);
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
