#include "Graph/Graph.hpp"

#include <algorithm>
#include <string>
#include <vector>
using namespace std;

int main() {
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
  cout << graph.areLayerNext(a1, a2) << "\n";
  cout << graph.areLayerNext(a2, a1) << "\n";
  cout << graph.areLayerNext(a1, a3) << "\n";
  return 0;
}