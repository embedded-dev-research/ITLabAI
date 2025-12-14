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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto inputLayer = std::make_unique<InputLayer>();
  auto ewLayer = std::make_unique<EWLayer>();

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* inputLayer_ptr = inputLayer.get();
  Layer* ewLayer_ptr = ewLayer.get();

  graph.setInput(inputLayer_ptr, input);
  graph.makeConnection(inputLayer_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer_ptr, ewLayer_ptr);

  ASSERT_EQ(graph.areLayerNext(inputLayer_ptr, fcLayer_ptr), 1);
}

TEST(graph, check_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto inputLayer = std::make_unique<InputLayer>();
  auto ewLayer = std::make_unique<EWLayer>();
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* inputLayer_ptr = inputLayer.get();
  Layer* ewLayer_ptr = ewLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();

  graph.setInput(inputLayer_ptr, input);
  graph.makeConnection(inputLayer_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer_ptr, ewLayer_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.setOutput(fcLayer2_ptr, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer_ptr, fcLayer2_ptr), 1);
}

TEST(graph, check_connection_when_not_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto inputLayer = std::make_unique<InputLayer>();
  auto ewLayer = std::make_unique<EWLayer>();
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* inputLayer_ptr = inputLayer.get();
  Layer* ewLayer_ptr = ewLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();

  graph.setInput(inputLayer_ptr, input);
  graph.makeConnection(inputLayer_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.setOutput(fcLayer2_ptr, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer_ptr, ewLayer_ptr), false);

  graph.makeConnection(fcLayer_ptr, ewLayer_ptr);

  ASSERT_EQ(graph.areLayerNext(fcLayer_ptr, ewLayer_ptr), true);
}

TEST(graph, check_connection_when_not_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer_ptr, fcLayer_ptr), 0);
}

TEST(graph, check_connection_when_not_connection2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer2_ptr, fcLayer4_ptr), 0);
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

TEST(graph, add_owned_layer_null_check) {
  Graph graph;
  std::unique_ptr<Layer> null_layer = nullptr;
  graph.addOwnedLayer(std::move(null_layer));
  SUCCEED();
}

TEST(graph, make_connection_null_layers) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto valid_layer = std::make_unique<InputLayer>();
  Layer* valid_ptr = valid_layer.get();
  graph.addOwnedLayer(std::move(valid_layer));

  EXPECT_THROW(graph.makeConnection(nullptr, valid_ptr), std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(valid_ptr, nullptr), std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(nullptr, nullptr), std::invalid_argument);
}

TEST(graph, make_connection_same_layer) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto layer = std::make_unique<InputLayer>();
  Layer* layer_ptr = layer.get();
  graph.addOwnedLayer(std::move(layer));
  graph.setInput(layer_ptr, input);

  EXPECT_THROW(graph.makeConnection(layer_ptr, layer_ptr), std::out_of_range);
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

  auto input_layer = std::make_unique<InputLayer>();
  auto split_layer = std::make_unique<SplitLayer>(1, 2);
  auto ew_layer1 = std::make_unique<EWLayer>("relu");
  auto ew_layer2 = std::make_unique<EWLayer>("sigmoid");
  auto concat_layer = std::make_unique<ConcatLayer>(0);

  Layer* input_ptr = input_layer.get();
  Layer* split_ptr = split_layer.get();
  Layer* ew1_ptr = ew_layer1.get();
  Layer* ew2_ptr = ew_layer2.get();
  Layer* concat_ptr = concat_layer.get();
  graph.setSplitDistribution(split_dist);

  graph.addOwnedLayer(std::move(input_layer));
  graph.addOwnedLayer(std::move(split_layer));
  graph.addOwnedLayer(std::move(ew_layer1));
  graph.addOwnedLayer(std::move(ew_layer2));
  graph.addOwnedLayer(std::move(concat_layer));

  graph.setInput(input_ptr, input);
  graph.makeConnection(input_ptr, split_ptr);
  graph.makeConnection(split_ptr, ew1_ptr);
  graph.makeConnection(split_ptr, ew2_ptr);
  graph.makeConnection(ew1_ptr, concat_ptr);
  graph.makeConnection(ew2_ptr, concat_ptr);
  graph.setOutput(concat_ptr, output);

  ASSERT_TRUE(graph.areLayerNext(input_ptr, split_ptr));
  ASSERT_TRUE(graph.areLayerNext(split_ptr, ew1_ptr));
  ASSERT_TRUE(graph.areLayerNext(split_ptr, ew2_ptr));
  ASSERT_TRUE(graph.areLayerNext(ew1_ptr, concat_ptr));
  ASSERT_TRUE(graph.areLayerNext(ew2_ptr, concat_ptr));
}

TEST(graph, vertex_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_ANY_THROW(static_cast<void>(graph.getVertexValue(5)));
}

TEST(graph, edges_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_ANY_THROW(static_cast<void>(graph.getEdgeValue(999)));
}

TEST(graph, inputs_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_ANY_THROW(static_cast<void>(graph.getInputsSize(999)));
}

TEST(graph, get_layer_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  ASSERT_ANY_THROW(static_cast<void>(graph.getLayerFromID(999)));
}

TEST(graph_transformations, check_subgraphs_search) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  Graph subgraph;

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);

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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);
  auto ewLayer5 = std::make_unique<EWLayer>("relu");

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();
  Layer* ewLayer5_ptr = ewLayer5.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.makeConnection(fcLayer4_ptr, ewLayer5_ptr);
  graph.setOutput(ewLayer5_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, ewLayer5_ptr);

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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer3_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer3_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer3_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer3_ptr, fcLayer_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.setOutput(fcLayer4_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

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

  auto fcLayer = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_unique<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_unique<FCLayer>(weights, bias);
  auto ewLayer5 = std::make_unique<EWLayer>("relu");

  Layer* fcLayer_ptr = fcLayer.get();
  Layer* fcLayer2_ptr = fcLayer2.get();
  Layer* fcLayer3_ptr = fcLayer3.get();
  Layer* fcLayer4_ptr = fcLayer4.get();
  Layer* ewLayer5_ptr = ewLayer5.get();

  graph.setInput(fcLayer_ptr, input);
  graph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  graph.makeConnection(fcLayer_ptr, fcLayer4_ptr);
  graph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);
  graph.makeConnection(fcLayer4_ptr, ewLayer5_ptr);
  graph.setOutput(ewLayer5_ptr, output);

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.addSingleLayer(fcLayer3_ptr);
  subgraph.makeConnection(fcLayer3_ptr, ewLayer5_ptr);

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

  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<Layer*> layer_ptrs;

  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_unique<FCLayer>(weights, bias));
    layer_ptrs.push_back(layers.back().get());
  }
  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_unique<EWLayer>("relu"));
    layer_ptrs.push_back(layers.back().get());
  }

  graph.setInput(layer_ptrs[0], input);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> first_dist(0, num_vertices - 2);
  std::uniform_int_distribution<int> second_dist(1, num_vertices - 1);
  for (int i = 0; i < num_vertices; i++) {
    int rFirst = first_dist(rng);
    int rSecond = second_dist(rng);
    if ((rFirst == rSecond) ||
        ((layer_ptrs[rFirst]->getID() == layer_ptrs[rSecond]->getID()) &&
         (layer_ptrs[rFirst]->getID() != 0))) {
      continue;
    }
    if ((layer_ptrs[rFirst]->getID() >= graph.getLayersCount()) ||
        (rFirst != 0 && layer_ptrs[rFirst]->getID() == 0)) {
      graph.addSingleLayer(layer_ptrs[rFirst]);
    }
    graph.makeConnection(layer_ptrs[rFirst], layer_ptrs[rSecond]);
  }
  graph.setOutput(layer_ptrs[num_vertices - 1], output);

  for (auto& layer : layers) {
    graph.addOwnedLayer(std::move(layer));
  }

  subgraph.setInput(layer_ptrs[0], input);
  subgraph.makeConnection(layer_ptrs[0], layer_ptrs[50]);
  subgraph.makeConnection(layer_ptrs[50], layer_ptrs[1]);

  std::vector<std::vector<int>> res1 = find_subgraphs(graph, subgraph);
  double res1_time =
      elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph, subgraph);
  std::cerr << "Find subgraphs time in ms " << res1_time << '\n';
}
