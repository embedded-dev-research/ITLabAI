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

  graph.addOwnedLayer(std::move(inputLayer));
  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(ewLayer));

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

  graph.addOwnedLayer(std::move(inputLayer));
  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(ewLayer));
  graph.addOwnedLayer(std::move(fcLayer2));

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

  graph.addOwnedLayer(std::move(inputLayer));
  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(ewLayer));
  graph.addOwnedLayer(std::move(fcLayer2));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  ASSERT_EQ(graph.areLayerNext(fcLayer2_ptr, fcLayer4_ptr), 0);
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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  ASSERT_ANY_THROW(graph.getVertexValue(5));
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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  ASSERT_ANY_THROW(graph.getEdgeValue(999));
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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  ASSERT_ANY_THROW(graph.getInputsSize(999));
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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  ASSERT_ANY_THROW(graph.getLayerFromID(999));
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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));
  graph.addOwnedLayer(std::move(ewLayer5));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, ewLayer5_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<EWLayer>("relu"));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.makeConnection(fcLayer2_ptr, fcLayer3_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));

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

  graph.addOwnedLayer(std::move(fcLayer));
  graph.addOwnedLayer(std::move(fcLayer2));
  graph.addOwnedLayer(std::move(fcLayer3));
  graph.addOwnedLayer(std::move(fcLayer4));
  graph.addOwnedLayer(std::move(ewLayer5));

  subgraph.setInput(fcLayer_ptr, input);
  subgraph.makeConnection(fcLayer_ptr, fcLayer2_ptr);
  subgraph.addSingleLayer(fcLayer3_ptr);
  subgraph.makeConnection(fcLayer3_ptr, ewLayer5_ptr);

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<EWLayer>("relu"));

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
  for (int i = 0; i < num_vertices; i++) {
    int rFirst = rand() % (num_vertices - 1);
    int rSecond = 1 + rand() % (num_vertices - 1);
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

  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));
  subgraph.addOwnedLayer(std::make_unique<FCLayer>(weights, bias));

  std::vector<std::vector<int>> res1 = find_subgraphs(graph, subgraph);
  double res1_time =
      elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph, subgraph);
  std::cerr << "Find subgraphs time in ms " << res1_time << std::endl;
}