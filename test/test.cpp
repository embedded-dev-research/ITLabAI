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
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.conection(a1, a2);
  graph.conection(a2, a3);
  graph.conection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_conection1) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.conection(a1, a2);
  graph.conection(a2, a3);
  graph.conection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_conection_when_not_conection) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.conection(a1, a2);
  graph.conection(a2, a3);
  graph.conection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_conection_when_not_conection2) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.conection(a1, a2);
  graph.conection(a2, a3);
  graph.conection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_conection_when_not_conection3) {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.conection(a1, a2);
  graph.conection(a2, a3);
  graph.conection(a1, a4);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}
