#include "Graph/Graph.hpp"

#include <algorithm>
#include <string>
#include <vector>

int main() {
  const std::vector<int> vec;
  Graph graph(5);
  LayerExample a1(0, 1);
  LayerExample a2(1, 2);
  LayerExample a3(2, 1);
  LayerExample a4(3, 2);
  graph.setInput(a1, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  graph.areLayerNext(a1, a2);
  graph.areLayerNext(a2, a1);
  graph.areLayerNext(a1, a3);
  return 0;
}
