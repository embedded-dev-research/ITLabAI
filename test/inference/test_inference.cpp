#include <vector>

#include "graph/graph.hpp"
#include "gtest/gtest.h"

using namespace itlab_2023;

TEST(bfs, check_result_vec) {
  const std::vector<int> vec_in = {1, 2, 3, 4};
  std::vector<int> vec_out(4);
  Graph graph(5);
  LayerExample a1(kInput);
  LayerExample a2(kFullyConnected);
  LayerExample a3(kFullyConnected);
  LayerExample a4(kDropout);
  LayerExample a5(kOutput);
  graph.setInput(a1, vec_in);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a1, a3);
  graph.makeConnection(a2, a4);
  graph.makeConnection(a4, a5);
  graph.setOutput(a5, vec_out);
  graph.inference();
  ASSERT_EQ(vec_in, vec_out);
}
