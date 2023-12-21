#pragma once
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

class Layer {
 private:
  int id;
  std::string name;
  int type;
  std::string version;
  int numInputs;
  int numNeurons;
  std::vector<int> primer;

 public:
  Layer(int id1, int type1) : id(id1), type(type1) {}
  int checkID() const { return id; }
  void In(const std::vector<int>& a) { primer = a; }
  std::vector<int> Out() { return primer; }
};

class Graph {
  int BiggestSize;
  int V;
  std::vector<Layer> layers;
  std::vector<int> arrayV;
  std::vector<int> arrayE;
  int start;
  int end;

 public:
  Graph(int vertices) : BiggestSize(vertices) {
    if (BiggestSize < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV.push_back(0);
    V = 0;
  }
  void setInput(const Layer& lay, const std::vector<int>& vec) {
    layers.push_back(lay);
    arrayV.push_back(0);
    start = lay.checkID();
    V++;
  }
  void makeConnection(const Layer& layPrev, const Layer& layNext) {
    layers.push_back(layNext);
    arrayV[V] = arrayV[V - 1];
    arrayV.push_back(arrayE.size());
    if (layPrev.checkID() == layNext.checkID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV.size() - layPrev.checkID() - 1; ind++)
      arrayV[layPrev.checkID() + ind]++;
    arrayE.insert(arrayE.begin() + arrayV[layPrev.checkID()],
                  layNext.checkID());
    V++;
    arrayV[V] = arrayE.size();
  }
  bool areLayerNext(const Layer& layPrev, const Layer& layNext) {
    for (int i = arrayV[layPrev.checkID()]; i < arrayV[layPrev.checkID() + 1];
         i++) {
      if (arrayE[i] == layNext.checkID()) {
        return true;
      }
    }
    return false;
  }
};
