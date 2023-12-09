#include "Graph/Graph.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  Graph graph(5);
  Layer a1;
  Layer a2;
  Layer a3;
  graph.addLayer(a1);
  graph.addLayer(a2);
  graph.addLayer(a3);
  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  graph.checkarrays();
  cout << graph.areLayerNext(0, 1) << "\n";
  cout << graph.areLayerNext(2, 0) << "\n";
  return 0;
}