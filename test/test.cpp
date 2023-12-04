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
  Graph graph(1);
  Node a("Convolution");
  graph.addNode(&a);
  Node b("Const");
  graph.addNode(&b);
  Node c("Parameter");
  graph.addNode(&c);
  graph.addEdge(&a, &b);
  ASSERT_EQ(graph.areNodesConnected(&a, &b), 1);
}
TEST(graph, check_next) {
  Graph graph(1);
  Node a("Convolution");
  graph.addNode(&a);
  Node b("Const");
  graph.addNode(&b);
  Node c("Parameter");
  graph.addNode(&c);
  graph.addEdge(&a, &b);
  ASSERT_EQ(graph.areNodeNext(&a, &b), 1);
}
TEST(graph, check_prev) {
  Graph graph(1);
  Node a("Convolution");
  graph.addNode(&a);
  Node b("Const");
  graph.addNode(&b);
  Node c("Parameter");
  graph.addNode(&c);
  graph.addEdge(&a, &b);
  ASSERT_EQ(graph.areNodePrev(&a, &b), 0);
}
TEST(graph, check_conection_when_not_conection) {
  Graph graph(1);
  Node a("Convolution");
  graph.addNode(&a);
  Node b("Const");
  graph.addNode(&b);
  Node c("Parameter");
  graph.addNode(&c);
  graph.addEdge(&a, &b);
  ASSERT_EQ(graph.areNodesConnected(&a, &c), 0);
}
TEST(graph, check_conection_when_not_conection2) {
  Graph graph(1);
  Node a("Convolution");
  graph.addNode(&a);
  Node b("Const");
  graph.addNode(&b);
  Node c("Parameter");
  graph.addNode(&c);
  graph.addEdge(&a, &b);
  ASSERT_EQ(graph.areNodesConnected(&c, &b), 0);
}
