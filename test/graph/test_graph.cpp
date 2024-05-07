#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"

using namespace itlab_2023;

TEST(graph, check_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> vec2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Graph graph(5);
  FCLayer a1;
  InputLayer a2;
  EWLayer a3;
  graph.setInput(a1, bias);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  ASSERT_EQ(graph.areLayerNext(a1, a2), 1);
}
TEST(graph, check_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> vec2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Graph graph(5);
  FCLayer a1;
  InputLayer a2;
  EWLayer a3;
  FCLayer a4;
  graph.setInput(a1, bias);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, bias);
  ASSERT_EQ(graph.areLayerNext(a1, a4), 1);
}
TEST(graph, check_connection_when_not_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> vec2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Graph graph(5);
  FCLayer a1;
  InputLayer a2;
  EWLayer a3;
  FCLayer a4;
  graph.setInput(a1, bias);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, bias);
  ASSERT_EQ(graph.areLayerNext(a1, a3), 0);
}
TEST(graph, check_connection_when_not_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> vec2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Graph graph(5);
  FCLayer a1;
  FCLayer a2;
  FCLayer a3;
  FCLayer a4;
  graph.setInput(a1, bias);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, bias);
  ASSERT_EQ(graph.areLayerNext(a1, a1), 0);
}
TEST(graph, check_connection_when_not_connection2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  const std::vector<float> vec2 = {9.0F, 6.4F, 17.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor output;
  Shape wshape({3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Graph graph(5);
  FCLayer a1;
  FCLayer a2;
  FCLayer a3;
  FCLayer a4;
  graph.setInput(a1, bias);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.setOutput(a4, bias);
  ASSERT_EQ(graph.areLayerNext(a2, a4), 0);
}
