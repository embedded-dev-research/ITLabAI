#include <algorithm>
#include <random>
#include <vector>

#include "graph/graph.hpp"
#include "graph_transformations/graph_transformations.hpp"
#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

TEST(graph, check_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 1);
}

TEST(graph, check_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.setOutput(fcLayer2, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, fcLayer2), 1);
}

TEST(graph, check_connection_when_not_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.setOutput(fcLayer2, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), false);

  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), true);
}

TEST(graph, check_connection_when_not_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, fcLayer), 0);
}

TEST(graph, check_connection_when_not_connection2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer2, fcLayer4), 0);
}

TEST(graph, vertex_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(graph.getVertexValue(5));
}

TEST(graph, edges_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(graph.getEdgeValue(999));
}

TEST(graph, inputs_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(graph.getInputsSize(999));
}

TEST(graph, get_layer_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(graph.getLayerFromID(999));
}

TEST(graph_transformations, check_subgraphs_search) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, fcLayer2);
  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 2}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);
  auto ewLayer5 = std::make_shared<EWLayer>("relu");

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.makeConnection(fcLayer4, ewLayer5);
  graph.setOutput(ewLayer5, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, ewLayer5);
  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({3, 4}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer3, fcLayer);
  graph.makeConnection(fcLayer3, fcLayer4);
  graph.setOutput(fcLayer4, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, fcLayer2);
  subgraph.makeConnection(fcLayer2, fcLayer3);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({0, 1, 2}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search3) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer3, fcLayer);
  graph.makeConnection(fcLayer2, fcLayer4);
  graph.setOutput(fcLayer4, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, fcLayer2);
  subgraph.makeConnection(fcLayer2, fcLayer3);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({2, 0, 1}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search4) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer3, fcLayer);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, fcLayer2);
  subgraph.makeConnection(fcLayer2, fcLayer3);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 2, 0}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search5) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph(5);
  Graph subgraph(2);
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);
  auto ewLayer5 = std::make_shared<EWLayer>("relu");

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer4, ewLayer5);
  graph.setOutput(ewLayer5, output);

  subgraph.setInput(fcLayer, input);
  subgraph.makeConnection(fcLayer, fcLayer2);
  subgraph.addSingleLayer(fcLayer3);
  subgraph.makeConnection(fcLayer3, ewLayer5);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 3, 2, 4}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_big_random) {
  const int num_vertices = 1000;
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph(num_vertices);
  Graph subgraph(3);

  std::vector<std::shared_ptr<Layer>> layers;
  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_shared<FCLayer>(weights, bias));
  }
  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_shared<EWLayer>("relu"));
  }

  graph.setInput(layers[0], input);
  for (int i = 0; i < num_vertices; i++) {
    int rFirst = rand() % (num_vertices - 1);
    int rSecond = 1 + rand() % (num_vertices - 1);
    if ((rFirst == rSecond) ||
        ((layers[rFirst]->getID() == layers[rSecond]->getID()) &&
         (layers[rFirst]->getID() != 0))) {
      continue;
    }
    if ((layers[rFirst]->getID() >= graph.getLayersCount()) ||
        (rFirst != 0 && layers[rFirst]->getID() == 0)) {
      graph.addSingleLayer(layers[rFirst]);
    }
    graph.makeConnection(layers[rFirst], layers[rSecond]);
  }
  graph.setOutput(layers[num_vertices - 1], output);

  subgraph.setInput(layers[0], input);
  subgraph.makeConnection(layers[0], layers[50]);
  subgraph.makeConnection(layers[50], layers[1]);

  std::vector<std::vector<int>> res1 = find_subgraphs(graph, subgraph);
  double res1_time =
      elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph, subgraph);
  std::cerr << "Find subgraphs time in ms " << res1_time << std::endl;
}
