#include <algorithm>
#include <random>
#include <vector>

#include "graph/graph.hpp"
#include "graph_transformations/graph_transformations.hpp"
#include "gtest/gtest.h"
#include "layers/ConcatLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/SplitLayer.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

TEST(graph, check_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 1);
}

TEST(graph, check_connection_remove) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeConnection(fcLayer->getID(), ewLayer->getID());
  graph.removeConnection(inputLayer->getID(), fcLayer->getID());

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), 0);
  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 0);
}

TEST(graph, check_connection_remove_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeConnection(999, -1));
}

TEST(graph, check_connection_remove_no_edge) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeConnection(0, 2));
}

TEST(graph, check_connection_double_remove_throw) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeConnection(fcLayer->getID(), ewLayer->getID());
  ASSERT_ANY_THROW(graph.removeConnection(fcLayer->getID(), ewLayer->getID()));
}

TEST(graph, check_layer_remove) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeSingleLayer(fcLayer->getID());

  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 0);
  ASSERT_ANY_THROW(graph.areLayerNext(fcLayer, ewLayer));
}

TEST(graph, check_layer_remove_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeSingleLayer(999));
  ASSERT_ANY_THROW(graph.removeSingleLayer(-1));
}

TEST(graph, check_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

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

  Graph graph;

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

  Graph graph;

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

  Graph graph;

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

TEST(graph, set_split_distribution) {
  Graph graph;
  std::vector<std::vector<std::pair<int, int>>> split_dist = {
      {{1, 0}, {2, 1}}, {{3, 0}, {4, 0}, {5, 1}}};
  graph.setSplitDistribution(split_dist);
  SUCCEED();
}

TEST(graph, set_input_null_layer) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  EXPECT_THROW(graph.setInput(nullptr, input), std::invalid_argument);
}

TEST(graph, make_connection_null_layers) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto valid_layer = std::make_shared<InputLayer>();

  EXPECT_THROW(graph.makeConnection(nullptr, valid_layer),
               std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(valid_layer, nullptr),
               std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(nullptr, nullptr), std::invalid_argument);
}

TEST(graph, make_connection_same_layer) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto layer = std::make_shared<InputLayer>();
  graph.setInput(layer, input);

  EXPECT_THROW(graph.makeConnection(layer, layer), std::out_of_range);
}

TEST(graph, set_output_null_layer) {
  Graph graph;
  Tensor output;
  EXPECT_THROW(graph.setOutput(nullptr, output), std::invalid_argument);
}

TEST(graph, get_vertex_value_invalid_id) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getVertexValue(1000)),
               std::invalid_argument);
}

TEST(graph, get_edge_value_invalid_pos) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getEdgeValue(1000)),
               std::invalid_argument);
}

TEST(graph, get_inputs_size_invalid_id) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getInputsSize(1000)),
               std::invalid_argument);
}

TEST(graph, get_layer_from_id_invalid_id) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getLayerFromID(1000)),
               std::invalid_argument);
}

TEST(graph, complex_graph_with_split_distribution) {
  std::vector<std::vector<std::pair<int, int>>> split_dist = {{{2, 0}, {3, 1}}};

  Graph graph(10, split_dist);
  Tensor input = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  Tensor output;

  auto input_layer = std::make_shared<InputLayer>();
  auto split_layer = std::make_shared<SplitLayer>(1, 2);
  auto ew_layer1 = std::make_shared<EWLayer>("relu");
  auto ew_layer2 = std::make_shared<EWLayer>("sigmoid");
  auto concat_layer = std::make_shared<ConcatLayer>(0);
  graph.setSplitDistribution(split_dist);

  graph.setInput(input_layer, input);
  graph.makeConnection(input_layer, split_layer);
  graph.makeConnection(split_layer, ew_layer1);
  graph.makeConnection(split_layer, ew_layer2);
  graph.makeConnection(ew_layer1, concat_layer);
  graph.makeConnection(ew_layer2, concat_layer);
  graph.setOutput(concat_layer, output);

  ASSERT_TRUE(graph.areLayerNext(input_layer, split_layer));
  ASSERT_TRUE(graph.areLayerNext(split_layer, ew_layer1));
  ASSERT_TRUE(graph.areLayerNext(split_layer, ew_layer2));
  ASSERT_TRUE(graph.areLayerNext(ew_layer1, concat_layer));
  ASSERT_TRUE(graph.areLayerNext(ew_layer2, concat_layer));
}

TEST(graph, vertex_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getVertexValue(5)));
}

TEST(graph, edges_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

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

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getInputsSize(999)));
}

TEST(graph, get_layer_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

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

TEST(graph, get_in_layers_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(graph.getInLayers(999));
}

TEST(graph, get_in_layers) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_NO_THROW(graph.getInLayers(0));
}

TEST(graph_transformations, check_subgraphs_search) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;

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

  Graph graph;
  Graph subgraph;
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

  Graph graph;
  Graph subgraph;
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

  Graph graph;
  Graph subgraph;
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

  Graph graph;
  Graph subgraph;
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

  Graph graph;
  Graph subgraph;
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
  Graph graph;
  Graph subgraph;

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

class SubgraphTestsParameterized
    : public ::testing::TestWithParam<std::vector<std::tuple<int, int>>> {};

TEST_P(SubgraphTestsParameterized, check_subgraphs_big_random_lines) {
  auto data = GetParam();
  for (size_t m = 0; m < data.size(); m++) {
    std::cerr << "(" << std::get<1>(data[m]) << ") ";
    int num_vertices = std::get<0>(data[m]);
    int num_vertices_sub = std::get<1>(data[m]);
    const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
    Tensor weights = make_tensor<float>(vec1, {3, 2});
    Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
    Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
    Tensor output;
    Graph graph;
    Graph subgraph;
    std::vector<std::shared_ptr<Layer>> layers;
    for (int i = 0; i < num_vertices; i++) {
      layers.push_back(std::make_shared<FCLayer>(weights, bias));
    }
    graph.setInput(layers[0], input);
    for (int i = 0; i < num_vertices - 1; i++) {
      graph.makeConnection(layers[i], layers[i + 1]);
    }
    graph.setOutput(layers[num_vertices - 1], output);

    std::shared_ptr<Layer> temp_layer =
        std::make_shared<FCLayer>(weights, bias);
    subgraph.setInput(temp_layer, input);
    std::shared_ptr<Layer> temp_layer2 =
        std::make_shared<FCLayer>(weights, bias);
    for (int i = 0; i < num_vertices_sub; i++) {
      subgraph.makeConnection(temp_layer, temp_layer2);
      temp_layer = temp_layer2;
      temp_layer2 = std::make_shared<FCLayer>(weights, bias);
    }

    double res1_time = elapsed_time_avg<double, std::milli>(1, find_subgraphs,
                                                            graph, subgraph);
    std::cerr << "Find subgraphs time in ms "
              << res1_time / (100 * num_vertices_sub * num_vertices_sub)
              << std::endl;
  }
}

std::vector<std::tuple<int, int>> genVector() {
  std::vector<std::tuple<int, int>> results(10);
  for (size_t i = 0; i < results.size(); i++) {
    results[i] = std::tuple<int, int>(105, 2 + 2 * static_cast<int>(i));
  }
  return results;
}

INSTANTIATE_TEST_SUITE_P(graph_transformations, SubgraphTestsParameterized,
                         ::testing::Values(genVector()));
