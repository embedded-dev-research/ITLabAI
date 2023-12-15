#include "Graph/Graph.hpp"

#include <algorithm>
#include <iostream>
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
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addLayer(a4);
  graph.addEdge(0, 1);
  graph.addEdge(1, 2);
  graph.addEdge(0, 3);
  graph.checkArrays();
  cout << graph.areLayerNext(0, 1) << "\n";
  cout << graph.areLayerNext(2, 0) << "\n";
  std::vector<int> VecForSearch = graph.BreadthFirstSearch(0, 2);
  std::vector<int> vecres = graph.TraversalGraph(vec, VecForSearch);
  cout << "\n";
  for (size_t i = 0; i < vecres.size(); ++i) {
    std::cout << vecres[i] << " ";
  }
  return 0;
}