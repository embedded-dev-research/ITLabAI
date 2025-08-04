#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"

using namespace it_lab_ai;

TEST(graph, check_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph(5);
  FCLayer fcLayer(weights, bias);
  InputLayer inputLayer;
  EWLayer ewLayer;

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
  FCLayer fcLayer(weights, bias);
  InputLayer inputLayer;
  EWLayer ewLayer;
  FCLayer fcLayer2(weights, bias);

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
  FCLayer fcLayer(weights, bias);
  InputLayer inputLayer;
  EWLayer ewLayer;
  FCLayer fcLayer2(weights, bias);

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
  FCLayer fcLayer(weights, bias);
  FCLayer fcLayer2(weights, bias);
  FCLayer fcLayer3(weights, bias);
  FCLayer fcLayer4(weights, bias);

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
  FCLayer fcLayer(weights, bias);
  FCLayer fcLayer2(weights, bias);
  FCLayer fcLayer3(weights, bias);
  FCLayer fcLayer4(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer2, fcLayer4), 0);
}
