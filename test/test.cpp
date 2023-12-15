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
TEST(graph, check_TraversalGraph1) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addLayer(a4);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(0, 3);
  std::vector<int> VecForSearch = graph.BreadthFirstSearch(0, 3);
  std::vector<int> vecres = graph.TraversalGraph(vec, VecForSearch);
  std::vector<int> vec1 = {4, 6, 8, 10};
  ASSERT_EQ(vecres, vec1);
}
TEST(graph, check_TraversalGraph2) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addLayer(a4);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(0, 3);
  std::vector<int> VecForSearch = graph.BreadthFirstSearch(0, 2);
  std::vector<int> vecres = graph.TraversalGraph(vec, VecForSearch);
  std::vector<int> vec1 = {5, 7, 9, 11};
  ASSERT_EQ(vecres, vec1);
}