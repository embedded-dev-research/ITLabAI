#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"

using namespace itlab_2023;

TEST(graph, check_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, vec_out);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_connection1) {
  const std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, vec_out);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_connection_when_not_connection) {
  const std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, vec_out);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_connection_when_not_connection2) {
  const std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, vec_out);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_connection_when_not_connection3) {
  const std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, vec_out);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}
