#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <unordered_map>

#include "build.hpp"
#include "graph_transformations/graph_transformations.hpp"
#include "layers_fused/ConvRelu.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

void alexnet_inf_careless(Graph& graph, const RuntimeOptions& options,
                          Tensor& input, Tensor& output) {
  auto* o = new Tensor(output);
  auto* i = new Tensor(input);
  graph.inference(options);
  graph.setOutput(*o);
  graph.setInput(*i);
}

void alexnet_comparison() {
  std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                891, 957,  1027, 973,  1008};
  size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  int count_pic = static_cast<int>(sum) + 10;
  std::vector<float> res(count_pic * 28 * 28, 1.0F);
  Tensor input;
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor output = make_tensor(vec, sh1);

  Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
  Tensor t = make_tensor<float>(res, sh);
  input = t;

  RuntimeOptions options;
  Graph graph;
  Graph graph2;
  build_graph_linear(graph, input, output, options, true);
  Graph subgraph;
  std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
  std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
  subgraph.setInput(layer_0, input);
  subgraph.makeConnection(layer_0, layer_1);
  std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
      std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
  changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
  Tensor input_c = input;
  Tensor output_c = output;
  double time1 = elapsed_time_avg<double, std::milli>(
      2, alexnet_inf_careless, graph, options, input_c, output_c);
  double time2 = elapsed_time_avg<double, std::milli>(
      2, alexnet_inf_careless, graph2, options, input_c, output_c);
  std::cout << time1 << " for unchanged graph\n";
  std::cout << time2 << " for convrelu graph\n";
}

int main() {
  int type = 2;
  Tensor input = make_tensor(std::vector<int>({0}));
  RuntimeOptions options;
  alexnet_comparison();
  if (type == 0) {
    Graph graph1;
    build_graph(graph1, input, input, MODEL_PATH_DENSENET_ONNX, options, false);

    Graph subgraph;
    Tensor scale = make_tensor(std::vector<float>({1.0F}));
    std::shared_ptr<Layer> layer_0 =
        std::make_shared<BatchNormalizationLayer>(scale, scale, scale, scale);
    std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_2 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_1, layer_2);
    subgraph.makeConnection(layer_2, layer_3);
    subgraph.makeConnection(layer_3, layer_4);

    Graph subgraph2;
    std::shared_ptr<Layer> layer_5 = std::make_shared<ConcatLayer>();
    std::shared_ptr<Layer> layer_6 =
        std::make_shared<PoolingLayer>(Shape({1, 1, 1}), "max");
    std::shared_ptr<Layer> layer_7 = std::make_shared<ConvolutionalLayer>();
    subgraph2.setInput(layer_6, input);
    subgraph2.makeConnection(layer_6, layer_5);
    subgraph2.addSingleLayer(layer_7);
    subgraph2.makeConnection(layer_7, layer_5);

    auto vec = find_subgraphs(graph1, subgraph);
    auto vec2 = find_subgraphs(graph1, subgraph2);
  } else if (type == 1) {
    Graph graph1;
    build_graph(graph1, input, input, MODEL_PATH_RESNET_ONNX, options, false);

    Graph subgraph;
    std::shared_ptr<Layer> layer_0 = std::make_shared<TransposeLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<SoftmaxLayer>();
    std::shared_ptr<Layer> layer_2 = std::make_shared<ReshapeLayer>();
    std::shared_ptr<Layer> layer_3 = std::make_shared<ReshapeLayer>();
    std::shared_ptr<Layer> layer_4 = std::make_shared<ReshapeLayer>();
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_1, layer_2);
    subgraph.makeConnection(layer_2, layer_3);
    subgraph.makeConnection(layer_3, layer_4);

    auto vec = find_subgraphs(graph1, subgraph);
  } else if (type == 2) {
    Graph graph1;
    build_graph(graph1, input, input, MODEL_PATH_GOOGLENET_ONNX, options,
                false);

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
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_0, layer_4);
    subgraph.makeConnection(layer_0, layer_5);
    subgraph.makeConnection(layer_0, layer_6);
    subgraph.makeConnection(layer_4, layer_2);
    subgraph.makeConnection(layer_5, layer_3);

    auto vec = find_subgraphs(graph1, subgraph);
  }
  return 0;
}
