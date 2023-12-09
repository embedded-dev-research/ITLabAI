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
TEST(graph, check_conection) {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  ASSERT_EQ(graph.areLayerNext(0, 1), 1);
}
TEST(graph, check_conection1) {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  ASSERT_EQ(graph.areLayerNext(0, 2), 1);
}
TEST(graph, check_conection_when_not_conection) {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  ASSERT_EQ(graph.areLayerNext(1, 0), 0);
}
TEST(graph, check_conection_when_not_conection2) {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  ASSERT_EQ(graph.areLayerNext(1, 1), 0);
}
TEST(graph, check_conection_when_not_conection3) {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  ASSERT_EQ(graph.areLayerNext(2, 1), 0);
}
