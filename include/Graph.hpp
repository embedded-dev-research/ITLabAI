#pragma once
#include <iostream>
#include <vector>

class Node {
 public:
  int abc;
  std::vector<int> ConnectedNode;
  Node(int a = 0) { abc = a; }
  void addconnect(int connectedGraph) {
    ConnectedNode.push_back(connectedGraph);
  }
  std::vector<int> CheckConnectedNode() { return ConnectedNode; }
};

class Graph {
 public:
  int V;
  std::vector<Node> adjList;
  Graph(int vertices) : V(vertices) { adjList.resize(V); }
  void addEdge(int a, int b) { adjList[a].addconnect(b); }
  bool checkconnect(int a, int b) {
    std::vector<int> vec = adjList[a].CheckConnectedNode();
    bool found = false;
    for (const auto& element : vec) {
      if (element == b) {
        found = true;
        break;
      }
    }
    return found;
  }
};
