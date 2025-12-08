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
  int type = 2;
  Tensor aaaa = make_tensor(std::vector<int>({0}));
  if (type == 0) {
    Graph graph1;
    build_graph(graph1, aaaa, aaaa, MODEL_PATH_DENSENET_ONNX, false);

    Graph subgraph;
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

    Graph subgraph2;
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
    auto time2 = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                      graph1, subgraph2);
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
  } else if (type == 1) {
    Graph graph1;
    build_graph(graph1, aaaa, aaaa, MODEL_PATH_RESNET_ONNX, false);

    Graph subgraph;
    std::shared_ptr<Layer> layer_0 = std::make_shared<TransposeLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<SoftmaxLayer>();
    std::shared_ptr<Layer> layer_2 = std::make_shared<ReshapeLayer>();
    std::shared_ptr<Layer> layer_3 = std::make_shared<ReshapeLayer>();
    std::shared_ptr<Layer> layer_4 = std::make_shared<ReshapeLayer>();
    subgraph.setInput(layer_0, aaaa);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_1, layer_2);
    subgraph.makeConnection(layer_2, layer_3);
    subgraph.makeConnection(layer_3, layer_4);

    auto vec = find_subgraphs(graph1, subgraph);
    auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph1,
                                                     subgraph);
    for (int i = 0; i < vec.size(); i++) {
      for (int j = 0; j < vec[i].size(); j++) {
        std::cerr << vec[i][j] << ' ';
      }
      std::cerr << '\n';
    }
    std::cerr << "Time for path5:" << time << std::endl;
    return 0;
  } else if (type == 2) {
    Graph graph1;
    build_graph(graph1, aaaa, aaaa, MODEL_PATH_GOOGLENET_ONNX, false);

    Graph subgraph;
    Shape shape(2);
    std::shared_ptr<Layer> layer_0 = std::make_shared<ConcatLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_2 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_5 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_6 =
        std::make_shared<PoolingLayer>(shape, "max");
    subgraph.setInput(layer_0, aaaa);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_0, layer_4);
    subgraph.makeConnection(layer_0, layer_5);
    subgraph.makeConnection(layer_0, layer_6);
    subgraph.makeConnection(layer_4, layer_2);
    subgraph.makeConnection(layer_5, layer_3);

    auto vec = find_subgraphs(graph1, subgraph);
    auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph1,
                                                     subgraph);
    for (int i = 0; i < vec.size(); i++) {
      for (int j = 0; j < vec[i].size(); j++) {
        std::cerr << vec[i][j] << ' ';
      }
      std::cerr << '\n';
    }
    std::cerr << "Time for concat:" << time << std::endl;
    return 0;
  }
}
