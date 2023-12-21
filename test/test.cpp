#include "Graph/Graph.hpp"
#include "gtest/gtest.h"

TEST(basic, basic_test) {
  // Arrange
  int a = 2;
  int b = 3;

  // Act
  int c = a + b;

  // Assert
  ASSERT_EQ(5, c);
}
TEST(graph, check_connection) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_connection1) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_connection_when_not_connection) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_connection_when_not_connection2) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_connection_when_not_connection3) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}
