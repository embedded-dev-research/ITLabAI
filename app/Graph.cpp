#include "Graph/Graph.hpp"

#include <algorithm>
#include <string>
#include <vector>

int main() {
  std::vector<int> vec = {1, 2, 3, 4};
  Graph graph(5);
  Layer a1(0, 1);
  Layer a2(1, 2);
  Layer a3(2, 1);
  Layer a4(3, 2);
  graph.input(a1, 3, vec);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a1, a4);
  std::cout << graph.areLayerNext(a1, a2) << "\n";
  std::cout << graph.areLayerNext(a2, a1) << "\n";
  std::cout << graph.areLayerNext(a1, a3) << "\n";
  return 0;
}
