#pragma once
#include <iostream>
#include <string>
#include <vector>

class Node {
 public:
  int abc;
  std::vector<Node*> prevNodes;
  std::vector<Node*> nextNodes;
  std::string Type;
  Node(std::string type) : Type(type) {}
  void addPrev(Node* node) { prevNodes.push_back(node); }
  void addNext(Node* node) { nextNodes.push_back(node); }
  bool isConnected(Node* otherNode) {
    for (Node* node : nextNodes) {
      if (node == otherNode) {
        return true;
      }
    }
    for (Node* node : prevNodes) {
      if (node == otherNode) {
        return true;
      }
    }
    return false;
  }
  bool isNext(Node* otherNode) {
    for (Node* node : nextNodes) {
      if (node == otherNode) {
        return true;
      }
    }
    return false;
  }
  bool isPrev(Node* otherNode) {
    for (Node* node : prevNodes) {
      if (node == otherNode) {
        return true;
      }
    }
    return false;
  }
};

class Graph {
 public:
  int V;
  std::vector<Node*> adjList;
  Graph(int vertices) : V(vertices) {
    if (V < 0) {
      throw "out_of_range";
    }
    adjList.resize(V);
  }
  void addEdge(Node* a, Node* b) { 
    a->addNext(b);
    b->addPrev(a);
  }
  void addNode(Node* node) {
    adjList.push_back(node);
    V++;
  }
  bool areNodesConnected(Node* node1, Node* node2) { 
    return node1->isConnected(node2);
  }
  bool areNodeNext(Node* node1, Node* node2) { return node1->isNext(node2); }
  bool areNodePrev(Node* node1, Node* node2) { return node1->isPrev(node2); }
};
